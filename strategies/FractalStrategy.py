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
from technical import indicators, qtpylib


class FractalStrategy(IStrategy):
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

    # Signal timeframe for the strategy - using 15m as primary trend
    timeframe = "5m"  # Renamed from signal_timeframe
    primary_timeframe = "15m" # This can remain for your internal logic if needed
    major_timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "240": 0.12,   # After 240 minutes, exit at 12% profit
        "1440": 0.04,  # After 24 hours, exit at 4% profit
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.02

    # Trailing stoploss to lock in profits as trend continues
    trailing_stop = False
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

    # Laguerre RSI parameters
    laguerre_gamma = DecimalParameter(0.5, 0.9, default=0.7, decimals=1, space="buy", load=True, optimize=True)
    buy_laguerre_level = DecimalParameter(0.1, 0.4, default=0.2, decimals=1, space="buy", load=True, optimize=True)
    sell_laguerre_level = DecimalParameter(0.6, 0.9, default=0.8, decimals=1, space="sell", load=True, optimize=True) # For short entry, cross below this
    exit_long_laguerre_level = DecimalParameter(0.6, 0.9, default=0.8, decimals=1, space="sell", load=True, optimize=True)
    exit_short_laguerre_level = DecimalParameter(0.1, 0.4, default=0.2, decimals=1, space="buy", load=True, optimize=True)

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
        # Donchian Channels (using 5-period window)
        # These are used for trend identification
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=5).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=5).min()

        # Identify peaks: where donchian_upper equals the high from 3 periods ago
        dataframe['peak'] = np.where(
            dataframe['donchian_upper'] == dataframe['high'].shift(3),
            dataframe['donchian_upper'],
            np.nan
        )
        dataframe['peak'] = dataframe['peak'].ffill()

        # Identify troughs: where donchian_lower equals the low from 3 periods ago
        dataframe['trough'] = np.where(
            dataframe['donchian_lower'] == dataframe['low'].shift(3),
            dataframe['donchian_lower'],
            np.nan
        )
        dataframe['trough'] = dataframe['trough'].ffill()

        # --- Trend detection for peak (for higher_high and lower_high) and trough (for higher_low and lower_low) ---
        # Initialize temporary columns for trend direction
        # 0: flat, 1: rising, -1: falling
        dataframe['peak_trend_temp'] = 0
        dataframe.loc[dataframe['high'] > dataframe['peak'].shift(1), 'peak_trend_temp'] = 1
        dataframe.loc[dataframe['peak'] < dataframe['peak'].shift(1), 'peak_trend_temp'] = -1
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['peak_trend_temp'] = dataframe['peak_trend_temp'].replace(0, pd.NA).ffill()
        # higher_high is True if the prevailing trend of donchian_upper is upwards (1)
        dataframe['higher_high'] = (dataframe['peak_trend_temp'] == 1).fillna(False)
        # lower_high is True if the prevailing trend of donchian_upper is downwards (-1)
        dataframe['lower_high'] = (dataframe['peak_trend_temp'] == -1).fillna(False)

        dataframe['trough_trend_temp'] = 0
        dataframe.loc[dataframe['trough'] > dataframe['trough'].shift(1), 'trough_trend_temp'] = 1
        dataframe.loc[dataframe['low'] < dataframe['trough'].shift(1), 'trough_trend_temp'] = -1
        # 0: flat, 1: rising, -1: falling
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['trough_trend_temp'] = dataframe['trough_trend_temp'].replace(0, pd.NA).ffill()

        # higher_low is True if the prevailing trend of donchian_lower is upwards (1)
        dataframe['higher_low'] = (dataframe['trough_trend_temp'] == 1).fillna(False)
        # lower_low is True if the prevailing trend of donchian_lower is downwards (-1)
        dataframe['lower_low'] = (dataframe['trough_trend_temp'] == -1).fillna(False)

        # Note: You might want to drop the temporary columns if they are not used elsewhere:
        dataframe.drop(['peak_trend_temp', 'trough_trend_temp'], axis=1, inplace=True)

        # Choppiness Index
        dataframe['chop'] = pta.chop(dataframe['high'], dataframe['low'], dataframe['close'], length=14)

        return dataframe

    @informative('1h')
    def populate_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for major trend confirmation on 1h timeframe
        """
        # Donchian Channels (using 5-period window)
        # These are used for trend identification
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=5).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=5).min()

        # Identify peaks: where donchian_upper equals the high from 3 periods ago
        dataframe['peak'] = np.where(
            dataframe['donchian_upper'] == dataframe['high'].shift(3),
            dataframe['donchian_upper'],
            np.nan
        )
        dataframe['peak'] = dataframe['peak'].ffill()

        # Identify troughs: where donchian_lower equals the low from 3 periods ago
        dataframe['trough'] = np.where(
            dataframe['donchian_lower'] == dataframe['low'].shift(3),
            dataframe['donchian_lower'],
            np.nan
        )
        dataframe['trough'] = dataframe['trough'].ffill()

        # --- Trend detection for peak (for higher_high and lower_high) and trough (for higher_low and lower_low) ---
        # Initialize temporary columns for trend direction
        # 0: flat, 1: rising, -1: falling
        dataframe['peak_trend_temp'] = 0
        dataframe.loc[dataframe['high'] > dataframe['peak'].shift(1), 'peak_trend_temp'] = 1
        dataframe.loc[dataframe['peak'] < dataframe['peak'].shift(1), 'peak_trend_temp'] = -1
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['peak_trend_temp'] = dataframe['peak_trend_temp'].replace(0, pd.NA).ffill()
        # higher_high is True if the prevailing trend of donchian_upper is upwards (1)
        dataframe['higher_high'] = (dataframe['peak_trend_temp'] == 1).fillna(False)
        # lower_high is True if the prevailing trend of donchian_upper is downwards (-1)
        dataframe['lower_high'] = (dataframe['peak_trend_temp'] == -1).fillna(False)

        dataframe['trough_trend_temp'] = 0
        dataframe.loc[dataframe['trough'] > dataframe['trough'].shift(1), 'trough_trend_temp'] = 1
        dataframe.loc[dataframe['low'] < dataframe['trough'].shift(1), 'trough_trend_temp'] = -1
        # 0: flat, 1: rising, -1: falling
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['trough_trend_temp'] = dataframe['trough_trend_temp'].replace(0, pd.NA).ffill()

        # higher_low is True if the prevailing trend of donchian_lower is upwards (1)
        dataframe['higher_low'] = (dataframe['trough_trend_temp'] == 1).fillna(False)
        # lower_low is True if the prevailing trend of donchian_lower is downwards (-1)
        dataframe['lower_low'] = (dataframe['trough_trend_temp'] == -1).fillna(False)

        # Note: You might want to drop the temporary columns if they are not used elsewhere:
        dataframe.drop(['peak_trend_temp', 'trough_trend_temp'], axis=1, inplace=True)

        # Choppiness Index
        dataframe['chop'] = pta.chop(dataframe['high'], dataframe['low'], dataframe['close'], length=14)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds indicators for secondary trends and generates buy/sell signals
        """
        # Secondary trend indicators
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # Laguerre RSI
        dataframe['laguerre'] = indicators.laguerre(dataframe, gamma=self.laguerre_gamma.value)

        # Momentum and volume indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Volume confirmation
        dataframe['volume_mean'] = dataframe['volume'].rolling(10).mean()
        dataframe['volume_increased'] = dataframe['volume'] > (dataframe['volume_mean'] * self.volume_threshold.value)

        # Donchian Channels (using 30-period window)
        # These are used for trend identification
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=30).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=30).min()

        dataframe['stop_upper'] = dataframe['high'].rolling(window=10).max()
        dataframe['stop_lower'] = dataframe['low'].rolling(window=10).min()

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
                # Major trend is up (from 1h timeframe)
                # dataframe['major_uptrend_1h'] &
                # Volume confirms the movement
                # dataframe['volume_increased'] &
                (dataframe['volume'] > 0) &  # Ensure volume is not 0
                # Higher peak and higher trough condition
                (dataframe['higher_high_1h']) &
                (dataframe['higher_low_1h']) &
                # Enough energy on major and primary timeframe
                (dataframe['chop_1h'] > 40) &
                (dataframe['chop_15m'] > 45) &
                # Laguerre RSI confirmation
                (qtpylib.crossed_above(dataframe['laguerre'], self.buy_laguerre_level.value))
            ),
            'enter_long'] = 1

        # SHORT entries - Primary downtrend confirmed by secondary trend bounce
        if self.can_short:
            dataframe.loc[
                (
                    # Major trend is down (from 1h timeframe)
                    # dataframe['major_downtrend_1h'] &
                    # Volume confirms the movement
                    # dataframe['volume_increased'] &
                    (dataframe['volume'] > 0) &  # Ensure volume is not 0
                    # Add conditions based on peak and trough
                    (dataframe['lower_low_1h']) &
                    (dataframe['lower_high_1h']) &
                    # Enough energy on primary and major timeframe
                    (dataframe['chop_1h'] > 40) &
                    (dataframe['chop_15m'] > 45) &
                    # Laguerre RSI confirmation
                    (qtpylib.crossed_below(dataframe['laguerre'], self.sell_laguerre_level.value))
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
                # (dataframe['downtrend_15m_15m']) |
                # # Trend reversal on high volume
                # (
                #     dataframe['lower_high'] &
                #     dataframe['lower_low'] &
                #     dataframe['volume_increased']
                # ) |
                # Break below key support
                (dataframe['close'] < dataframe['trough_15m'])
                # Laguerre RSI exit condition
                # (qtpylib.crossed_below(dataframe['laguerre'], self.exit_long_laguerre_level.value))
            ),
            'exit_long'] = 1

        # Exit SHORT positions
        if self.can_short:
            dataframe.loc[
                (
                    # Reversal signals in primary trend
                    # (dataframe['uptrend_15m_15m']) |
                    # # Trend reversal on high volume
                    # (
                    #     dataframe['higher_high'] &
                    #     dataframe['higher_low'] &
                    #     dataframe['volume_increased']
                    # ) |
                    # Break above key resistance
                    (dataframe['close'] > dataframe['peak_15m'])
                    # Laguerre RSI exit condition
                    # (qtpylib.crossed_above(dataframe['laguerre'], self.exit_short_laguerre_level.value))
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