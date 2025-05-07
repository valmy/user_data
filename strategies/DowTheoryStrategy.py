# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class DowTheoryStrategy(IStrategy):
    """
    Strategy based on Dow Theory principles for trend detection.

    Key Dow Theory principles implemented:
    1. Primary trends identification using higher timeframe
    2. Secondary trend corrections using mid timeframe
    3. Confirmation through volume
    4. Trend continuation until valid reversal signals

    The strategy uses multiple timeframes to align with Dow Theory's
    concept of primary, secondary, and minor trends.
    """
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy - using 15m as primary trend
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.05,     # 5% profit is enough to exit
        "240": 0.03,   # After 240 minutes, exit at 3% profit
        "1440": 0.01,  # After 24 hours, exit at 1% profit
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.1

    # Trailing stoploss to lock in profits as trend continues
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    # Parameters for tuning
    trend_strength = IntParameter(20, 50, default=25, space="buy")
    volume_threshold = DecimalParameter(1.0, 3.0, default=1.5, space="buy")
    reversal_threshold = IntParameter(10, 30, default=15, space="sell")

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        We need higher timeframes for primary trend detection and secondary trend confirmations.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []

        # 15m timeframe for primary trend detection
        for pair in pairs:
            informative_pairs.append((pair, "15m"))

        # 1h timeframe for major trend confirmation
        for pair in pairs:
            informative_pairs.append((pair, "1h"))

        return informative_pairs

    @informative('15m')
    def populate_informative_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for primary trend identification on 15m timeframe
        """
        # Calculate primary trend indicators
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Primary trend determination
        dataframe['uptrend_15m'] = (dataframe['ema50'] > dataframe['ema200']) & (dataframe['close'] > dataframe['ema50'])
        dataframe['downtrend_15m'] = (dataframe['ema50'] < dataframe['ema200']) & (dataframe['close'] < dataframe['ema50'])

        # Relative strength
        dataframe['rsi_15m'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    @informative('1h')
    def populate_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for major trend confirmation on 1h timeframe
        """
        # Calculate major trend indicators
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)

        # Major trend determination (longer term)
        dataframe['major_uptrend'] = dataframe['sma50'] > dataframe['sma200']
        dataframe['major_downtrend'] = dataframe['sma50'] < dataframe['sma200']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds indicators for secondary trends and generates buy/sell signals
        """
        # Secondary trend indicators
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # Momentum and volume indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Volume confirmation
        dataframe['volume_mean'] = dataframe['volume'].rolling(10).mean()
        dataframe['volume_increased'] = dataframe['volume'] > (dataframe['volume_mean'] * self.volume_threshold.value)

        # Detect higher highs and higher lows for uptrend
        dataframe['higher_high'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        )

        dataframe['higher_low'] = (
            (dataframe['low'] > dataframe['low'].shift(1)) &
            (dataframe['low'].shift(1) > dataframe['low'].shift(2))
        )

        # Detect lower highs and lower lows for downtrend
        dataframe['lower_high'] = (
            (dataframe['high'] < dataframe['high'].shift(1)) &
            (dataframe['high'].shift(1) < dataframe['high'].shift(2))
        )

        dataframe['lower_low'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['low'].shift(1) < dataframe['low'].shift(2))
        )

        # Identify peak and trough
        dataframe['peak'] = ta.MAX(dataframe['high'], timeperiod=10)  # Local peak
        dataframe['trough'] = ta.MIN(dataframe['low'], timeperiod=10)  # Local trough

        # Donchian Channels peak and trough detection (using 10-period window)
        donchian_upper = dataframe['high'].rolling(window=10).max()
        donchian_lower = dataframe['low'].rolling(window=10).min()
        dataframe['donchian_peak'] = dataframe['high'] >= donchian_upper
        dataframe['donchian_trough'] = dataframe['low'] <= donchian_lower

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on Dow Theory principles, identify entry signals
        """
        # Default values
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # LONG entries - Primary uptrend confirmed by secondary trend bounce
        dataframe.loc[
            (
                # Primary trend is up (from 15m timeframe)
                dataframe['uptrend_15m_15m'] &
                # Major trend is up (from 1h timeframe)
                dataframe['major_uptrend_1h'] &
                # Secondary trend indicators
                (dataframe['ema20'] > dataframe['ema50']) &
                # Price bounced off support (secondary correction ended)
                (dataframe['low'].shift(1) < dataframe['ema50']) &
                (dataframe['close'] > dataframe['ema50']) &
                # Volume confirms the movement
                dataframe['volume_increased'] &
                # Pattern of higher lows (accumulation)
                dataframe['higher_low'] &
                # RSI showing strength but not overbought
                (dataframe['rsi'] > self.trend_strength.value) &
                (dataframe['rsi'] < 70) &
                (dataframe['volume'] > 0) &  # Ensure volume is not 0
                # Higher peak and higher trough condition
                (dataframe['peak'] > dataframe['peak'].shift(11)) &
                (dataframe['trough'] > dataframe['trough'].shift(11))
            ),
            'enter_long'] = 1

        # SHORT entries - Primary downtrend confirmed by secondary trend bounce
        if self.can_short:
            dataframe.loc[
                (
                    # Primary trend is down (from 15m timeframe)
                    dataframe['downtrend_15m_15m'] &
                    # Major trend is down (from 1h timeframe)
                    dataframe['major_downtrend_1h'] &
                    # Secondary trend indicators
                    (dataframe['ema20'] < dataframe['ema50']) &
                    # Price bounced off resistance (secondary correction ended)
                    (dataframe['high'].shift(1) > dataframe['ema50']) &
                    (dataframe['close'] < dataframe['ema50']) &
                    # Volume confirms the movement
                    dataframe['volume_increased'] &
                    # Pattern of lower highs (distribution)
                    dataframe['lower_high'] &
                    # RSI showing weakness but not oversold
                    (dataframe['rsi'] < (100 - self.trend_strength.value)) &
                    (dataframe['rsi'] > 30) &
                    (dataframe['volume'] > 0)  # Ensure volume is not 0
                    # Add conditions based on peak and trough
                    & (dataframe['peak'] < dataframe['peak'].shift(11)) &
                    (dataframe['trough'] < dataframe['trough'].shift(11))
                ),
                'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on Dow Theory principles, identify exit signals when trends reverse
        """
        # Default values
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Exit LONG positions
        dataframe.loc[
            (
                # Reversal signals in primary trend
                (dataframe['downtrend_15m_15m']) |
                # RSI overbought
                (dataframe['rsi'] > 80) |
                # Trend reversal on high volume
                (
                    dataframe['lower_high'] &
                    dataframe['lower_low'] &
                    dataframe['volume_increased']
                ) |
                # Break below key support
                (dataframe['close'] < dataframe['trough'].shift(1)) |
                # RSI divergence (falling RSI while price rising)
                (
                    (dataframe['close'] > dataframe['close'].shift(3)) &
                    (dataframe['rsi'] < dataframe['rsi'].shift(3)) &
                    (dataframe['rsi'] < self.reversal_threshold.value)
                )
            ),
            'exit_long'] = 1

        # Exit SHORT positions
        if self.can_short:
            dataframe.loc[
                (
                    # Reversal signals in primary trend
                    (dataframe['uptrend_15m_15m']) |
                    # RSI oversold
                    (dataframe['rsi'] < 20) |
                    # Trend reversal on high volume
                    (
                        dataframe['higher_high'] &
                        dataframe['higher_low'] &
                        dataframe['volume_increased']
                    ) |
                    # Break above key resistance
                    (dataframe['close'] > dataframe['peak'].shift(1)) |
                    # RSI divergence (rising RSI while price falling)
                    (
                        (dataframe['close'] < dataframe['close'].shift(3)) &
                        (dataframe['rsi'] > dataframe['rsi'].shift(3)) &
                        (dataframe['rsi'] > (100 - self.reversal_threshold.value))
                    )
                ),
                'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Limit leverage to reasonable values when using margin/futures
        """
        # Conservative leverage for Dow Theory - trend following is already strong
        return min(2.0, max_leverage)