# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
from freqtrade.ft_types.plot_annotation_type import AnnotationType

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

import logging
import traceback # Import traceback for detailed error logging
logger = logging.getLogger(__name__)

class FractalStrategy(IStrategy):
    """
    Strategy based on Fractal Energy principles by Doc Severson.

    Key Fractal Energy principles implemented:
    1. Fractal pattern recognition for market structure
    2. Energy accumulation and distribution cycles (Choppiness Index)
    3. Momentum confirmation through volume and price action
    4. Use of Laguerre RSI for entry signals
    5. Use contant risk per trade, let compounding profits

    The strategy uses multiple timeframes to identify fractal patterns
    and energy cycles across different market scales.
    """
    INTERFACE_VERSION = 3

    # Whether to use safe position adjustment
    position_adjustment_enable = True

    # Signal timeframe for the strategy - using 15m as primary trend
    timeframe = "5m"  # Renamed from signal_timeframe
    primary_timeframe = "15m" # This can remain for your internal logic if needed
    major_timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy
    minimal_roi = {
        # "240": 0.12,   # After 240 minutes, exit at 12% profit
        # "1440": 0.04,  # After 24 hours, exit at 4% profit
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.20

    # Trailing stoploss to lock in profits as trend continues
    trailing_stop = True
    trailing_stop_positive = 0.20
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = False

    signal_timeframe_minutes = timeframe_to_minutes(timeframe)
    primary_timeframe_minutes = timeframe_to_minutes(primary_timeframe)
    major_timeframe_minutes = timeframe_to_minutes(major_timeframe)

    # Calculate ratios
    if signal_timeframe_minutes == 0:
        ratio_primary_to_signal = float('inf') # Or handle as an error
    else:
        ratio_primary_to_signal = primary_timeframe_minutes / signal_timeframe_minutes

    if primary_timeframe_minutes == 0:
        ratio_major_to_primary = float('inf') # Or handle as an error
    else:
        ratio_major_to_primary = major_timeframe_minutes / primary_timeframe_minutes

    # ratio major to signal
    ratio_major_to_signal = major_timeframe_minutes / signal_timeframe_minutes


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = max(50, 3 * ratio_major_to_signal)

    # Parameters for tuning
    volume_threshold = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=False)

    # Laguerre RSI parameters
    laguerre_gamma = DecimalParameter(0.6, 0.8, default=0.68, decimals=2, space="buy", load=True, optimize=False)
    small_candle_ratio = DecimalParameter(1.0, 5.0, default=2.0, decimals=1, space="buy", load=True, optimize=True)
    buy_laguerre_level = DecimalParameter(0.1, 0.4, default=0.2, decimals=1, space="buy", load=True, optimize=False)
    sell_laguerre_level = DecimalParameter(0.6, 0.9, default=0.8, decimals=1, space="sell", load=True, optimize=False) # For short entry, cross below this

    # Choppiness Index parameters
    primary_chop_threshold = IntParameter(35, 60, default=40, space="buy", optimize=True)
    major_chop_threshold = IntParameter(35, 50, default=40, space="buy", optimize=True)

    # Custom trade size parameters
    max_risk_per_trade = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="buy", load=True, optimize=False)
    trailing_stop_ratio = DecimalParameter(3.0, 20.0, default=2.0, decimals=1, space="sell", load=True, optimize=True)

    test_compounding_mode: bool = True

    _force_leverage_one_for_this_trade: bool = False

    def is_backtest_mode(self) -> bool:
        """Check if the current run mode is backtest or hyperopt"""
        return self.dp.runmode.value in ["backtest", "hyperopt"]

    def get_total_equity(self):
        if self.is_backtest_mode() and not self.test_compounding_mode:
            # Get values from config, with defaults if not set
            ratio = self.config.get('tradable_balance_ratio', 1.0)
            wallet = self.config.get('dry_run_wallet', 1000)
            logger.debug(f"get_total_equity: Using config values. Ratio: {ratio}, Wallet: {wallet}")
            return ratio * wallet
        else:
            logger.debug(f"get_total_equity: Using live wallet balance: {self.wallets.get_total_stake_amount()}")
            return self.wallets.get_total_stake_amount()

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        We need higher timeframes for primary trend detection and major trend confirmations.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []

        # Primary timeframe for trend detection
        for pair in pairs:
            informative_pairs.append((pair, self.primary_timeframe))

        # Major timeframe for trend confirmation
        for pair in pairs:
            informative_pairs.append((pair, self.major_timeframe))

        return informative_pairs

    @informative(primary_timeframe)
    def populate_informative_primary(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for primary trend identification on primary_timeframe timeframe
        """
        # Donchian Channels (using 5-period window)
        # These are used for trend identification
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=5).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=5).min()

        # Identify peaks: where donchian_upper equals the high from 3 periods ago
        dataframe['peak'] = np.where(
            dataframe['donchian_upper'] == dataframe['high'].shift(2),
            dataframe['donchian_upper'],
            np.nan
        )
        dataframe['peak'] = dataframe['peak'].ffill()

        # Identify troughs: where donchian_lower equals the low from 3 periods ago
        dataframe['trough'] = np.where(
            dataframe['donchian_lower'] == dataframe['low'].shift(2),
            dataframe['donchian_lower'],
            np.nan
        )
        dataframe['trough'] = dataframe['trough'].ffill()

        # --- Trend detection for peak (for higher_high and lower_high) and trough (for higher_low and lower_low) ---
        # Initialize temporary columns for trend direction
        # 0: flat, 1: rising, -1: falling
        dataframe['peak_trend_temp'] = 0
        dataframe.loc[dataframe['high'] > dataframe['peak'].shift(1) * 1.001, 'peak_trend_temp'] = 1
        dataframe.loc[dataframe['peak'] > dataframe['peak'].shift(1) * 1.001, 'peak_trend_temp'] = 1
        dataframe.loc[dataframe['peak'] < dataframe['peak'].shift(1) * 0.999, 'peak_trend_temp'] = -1
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['peak_trend_temp'] = dataframe['peak_trend_temp'].replace(0, pd.NA).ffill()
        # higher_high is True if the prevailing trend of donchian_upper is upwards (1)
        dataframe['higher_high'] = (dataframe['peak_trend_temp'] == 1).fillna(False).astype(bool)
        # lower_high is True if the prevailing trend of donchian_upper is downwards (-1)
        dataframe['lower_high'] = (dataframe['peak_trend_temp'] == -1).fillna(False).astype(bool)

        dataframe['trough_trend_temp'] = 0
        dataframe.loc[dataframe['trough'] > dataframe['trough'].shift(1) * 1.001, 'trough_trend_temp'] = 1
        dataframe.loc[dataframe['low'] < dataframe['trough'].shift(1) * 0.999, 'trough_trend_temp'] = -1
        dataframe.loc[dataframe['trough'] < dataframe['trough'].shift(1) * 0.999, 'trough_trend_temp'] = -1
        # 0: flat, 1: rising, -1: falling
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['trough_trend_temp'] = dataframe['trough_trend_temp'].replace(0, pd.NA).ffill()

        # higher_low is True if the prevailing trend of donchian_lower is upwards (1)
        dataframe['higher_low'] = (dataframe['trough_trend_temp'] == 1).fillna(False).astype(bool)
        # lower_low is True if the prevailing trend of donchian_lower is downwards (-1)
        dataframe['lower_low'] = (dataframe['trough_trend_temp'] == -1).fillna(False).astype(bool)

        # Note: You might want to drop the temporary columns if they are not used elsewhere:
        dataframe.drop(['peak_trend_temp', 'trough_trend_temp'], axis=1, inplace=True)

        # Choppiness Index
        dataframe['chop'] = pta.chop(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
        return dataframe

    @informative(major_timeframe)
    def populate_informative_major(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for major trend confirmation on major_timeframe timeframe
        """

        # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        dataframe['ha_bullish'] = (heikinashi['close'] > heikinashi['open']).astype(bool)
        dataframe['ha_upswing'] = dataframe['ha_bullish'].rolling(4).sum() >= 3
        dataframe['ha_bearish'] = (heikinashi['close'] < heikinashi['open']).astype(bool)
        dataframe['ha_downswing'] = dataframe['ha_bearish'].rolling(4).sum() >= 3

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
        dataframe['higher_high'] = (dataframe['peak_trend_temp'] == 1).fillna(False).astype(bool)
        # lower_high is True if the prevailing trend of donchian_upper is downwards (-1)
        dataframe['lower_high'] = (dataframe['peak_trend_temp'] == -1).fillna(False).astype(bool)

        dataframe['trough_trend_temp'] = 0
        dataframe.loc[dataframe['trough'] > dataframe['trough'].shift(1), 'trough_trend_temp'] = 1
        dataframe.loc[dataframe['low'] < dataframe['trough'].shift(1), 'trough_trend_temp'] = -1
        # 0: flat, 1: rising, -1: falling
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe['trough_trend_temp'] = dataframe['trough_trend_temp'].replace(0, pd.NA).ffill()

        # higher_low is True if the prevailing trend of donchian_lower is upwards (1)
        dataframe['higher_low'] = (dataframe['trough_trend_temp'] == 1).fillna(False).astype(bool)
        # lower_low is True if the prevailing trend of donchian_lower is downwards (-1)
        dataframe['lower_low'] = (dataframe['trough_trend_temp'] == -1).fillna(False).astype(bool)

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
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Laguerre RSI
        dataframe['laguerre'] = indicators.laguerre(dataframe, gamma=self.laguerre_gamma.value)

        # Momentum and volume indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Volume confirmation
        dataframe['volume_mean'] = dataframe['volume'].rolling(10).mean()
        dataframe['volume_increased'] = dataframe['volume'] > (dataframe['volume_mean'] * self.volume_threshold.value)

        # Donchian Channels (using 30-period window)
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=30).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=30).min()

        dataframe['stop_upper'] = dataframe['high'].rolling(window=10).max()
        dataframe['stop_lower'] = dataframe['low'].rolling(window=10).min()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals with improved conditions, error handling, and optimizations
        """
        try:
            # Create a copy to avoid SettingWithCopyWarning
            df = dataframe.copy()
            # logger.info(f"DataFrame columns at start of populate_entry_trend: {df.columns.to_list()}")

            # Initialize signal columns
            df['enter_long'] = 0
            df['enter_short'] = 0

            # Calculate conditions with error handling
            try:
                # --- Debugging for entry condition error ---
                ptf_chop_col = f'chop_{self.primary_timeframe}'
                mtf_chop_col = f'chop_{self.major_timeframe}'

                if ptf_chop_col not in df.columns or mtf_chop_col not in df.columns:
                    logger.error(f"Chop columns missing! Primary: {ptf_chop_col in df.columns}, Major: {mtf_chop_col in df.columns}. All columns: {df.columns.to_list()}")
                    df['enter_long'] = 0
                    df['enter_short'] = 0
                    return df

                ptf_thresh_val = self.primary_chop_threshold.value
                mtf_thresh_val = self.major_chop_threshold.value

                # logger.debug(f"Primary chop ({ptf_chop_col}) dtype: {df[ptf_chop_col].dtype}, head: {df[ptf_chop_col].head(3).to_list()}, threshold: {ptf_thresh_val} (type: {type(ptf_thresh_val)})")
                # logger.debug(f"Major chop ({mtf_chop_col}) dtype: {df[mtf_chop_col].dtype}, head: {df[mtf_chop_col].head(3).to_list()}, threshold: {mtf_thresh_val} (type: {type(mtf_thresh_val)})")

                cond_ptf_chop = (df[ptf_chop_col] > ptf_thresh_val)
                cond_mtf_chop = (df[mtf_chop_col] > mtf_thresh_val)

                # logger.debug(f"cond_ptf_chop dtype: {cond_ptf_chop.dtype}, head: {cond_ptf_chop.head(3).to_list()}")
                # logger.debug(f"cond_mtf_chop dtype: {cond_mtf_chop.dtype}, head: {cond_mtf_chop.head(3).to_list()}")

                # --- Debugging ---
                # Get the ha_upswing from the informative major timeframe
                ha_upswing_col = f'ha_upswing_{self.major_timeframe}'
                if ha_upswing_col not in df.columns:
                    logger.error(f"ha_upswing column {ha_upswing_col} not found in dataframe columns: {df.columns.to_list()}")
                    return df

                # Get the ha_downswing from the informative major timeframe
                ha_downswing_col = f'ha_downswing_{self.major_timeframe}'
                if ha_downswing_col not in df.columns:
                    logger.error(f"ha_downswing column {ha_downswing_col} not found in dataframe columns: {df.columns.to_list()}")
                    return df

                # --- End Debugging ---

                # Pre-calculate common conditions for better performance
                df['strong_volume'] = df['volume'] > (df['volume_mean'] * 1.5)
                df['bullish_candle'] = df['close'] > df['open']
                df['bearish_candle'] = df['close'] < df['open']
                # df['above_ema20'] = df['close'] > df['ema20']
                back_range = int(3 * self.ratio_primary_to_signal)
                df['above_resistance'] = df['low'].rolling(window=back_range).min() >= df[f'trough_{self.primary_timeframe}']
                df['below_support'] = df['high'].rolling(window=back_range).max() <= df[f'peak_{self.primary_timeframe}']

                # Small candle condition: candle range must be smaller than small_candle_ratio * ATR
                df['candle_range'] = df['high'] - df['low']
                df['small_candle'] = df['candle_range'] < (self.small_candle_ratio.value * df['atr'])

                # LONG Entry Conditions
                long_condition = (
                    # signal: laguerre crosses above buy_laguerre_level
                    (qtpylib.crossed_above(dataframe['laguerre'], self.buy_laguerre_level.value)) &
                    # confirmation: strong volume
                    df['strong_volume'] &
                    # df['above_ema20'] &
                    df['above_resistance'] &
                    # small candle condition
                    df['small_candle'] &
                    # at least 2 of the last 3 major heikin ashi candles are bullish
                    df[ha_upswing_col] &
                    # enough energy (using pre-calculated conditions)
                    cond_ptf_chop &
                    cond_mtf_chop
                )

                # SHORT Entry Conditions
                short_condition = (
                    # signal: laguerre crosses below sell_laguerre_level
                    (qtpylib.crossed_below(dataframe['laguerre'], self.sell_laguerre_level.value)) &
                    # confirmation: strong volume
                    df['strong_volume'] &
                    # ~df['above_ema20'] &
                    df['below_support'] &
                    # small candle condition
                    df['small_candle'] &
                    # at least 2 of the last 3 major heikin ashi candles are bearish
                    df[ha_downswing_col] &
                    # enough energy
                    cond_ptf_chop &
                    cond_mtf_chop
                )

                # Apply conditions with position sizing
                df.loc[long_condition, 'enter_long'] = 1

                if self.can_short:
                    df.loc[short_condition, 'enter_short'] = 1

                # Limit the number of signals to avoid over-trading
                max_signals = len(df) // 30  # Max 1 signal per 30 candles

                # For long signals
                if sum(long_condition) > max_signals:
                    long_signals = df[long_condition].index[-max_signals:]
                    df['enter_long'] = 0
                    df.loc[long_signals, 'enter_long'] = 1

                # For short signals
                if self.can_short and sum(short_condition) > max_signals:
                    short_signals = df[short_condition].index[-max_signals:]
                    df['enter_short'] = 0
                    df.loc[short_signals, 'enter_short'] = 1

                # Debug info
                # logger.info(f"Generated {sum(df['enter_long'])} long and {sum(df['enter_short'])} short signals for {metadata['pair']}")

                return df

            except Exception as e:
                logger.error(f"Error in entry conditions for {metadata['pair']}: {str(e)}\n{traceback.format_exc()}")
                # Return dataframe with no signals if there's an error
                df['enter_long'] = 0
                df['enter_short'] = 0
                return df

        except Exception as e:
            logger.error(f"Critical error in populate_entry_trend for {metadata['pair']}: {str(e)}\n{traceback.format_exc()}")
            # Return the original dataframe with no signals if something goes wrong
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on trend reversals, profit targets, and stop losses
        """
        try:
            # Create a copy to avoid SettingWithCopyWarning
            df = dataframe.copy()

            # Initialize exit columns
            df['exit_long'] = 0
            df['exit_short'] = 0
            df['exit_reason'] = ''  # New column to store exit reason

            try:
                hh_col = f'higher_high_{self.primary_timeframe}'
                ll_col = f'lower_low_{self.primary_timeframe}'
                trough_col = f'trough_{self.primary_timeframe}'
                peak_col = f'peak_{self.primary_timeframe}'

                if not all(col in df.columns for col in [hh_col, ll_col, trough_col, peak_col]):
                    logger.error(f"Exit condition columns missing! HH: {hh_col in df.columns}, LL: {ll_col in df.columns}, Trough: {trough_col in df.columns}, Peak: {peak_col in df.columns}. All columns: {df.columns.to_list()}")
                    return df # Return df with no exits

                # Exit LONG positions
                exit_long_price_condition = (df['close'] < df[trough_col])
                exit_long_trend_condition = (df[ll_col].astype(bool))

                exit_long_condition = exit_long_price_condition | exit_long_trend_condition

                # Exit SHORT positions
                exit_short_price_condition = (df['close'] > df[peak_col])
                exit_short_trend_condition = (df[hh_col].astype(bool))

                exit_short_condition = exit_short_price_condition | exit_short_trend_condition

                # Apply exit conditions and set exit reason
                df.loc[exit_long_condition, 'exit_long'] = 1
                df.loc[exit_long_price_condition, 'exit_reason'] = 'price'
                df.loc[exit_long_trend_condition, 'exit_reason'] = 'trend'

                if self.can_short:
                    df.loc[exit_short_condition, 'exit_short'] = 1
                    df.loc[exit_short_price_condition, 'exit_reason'] = 'price'
                    df.loc[exit_short_trend_condition, 'exit_reason'] = 'trend'

                return df
            except Exception as e_inner:
                logger.error(f"Error in exit trend condition calculation for {metadata['pair']}: {str(e_inner)}\n{traceback.format_exc()}")
                # df['exit_long'] = 0 and df['exit_short'] = 0 are already set
                return df

        except Exception as e:
            logger.error(f"Critical error in populate_exit_trend for {metadata['pair']}: {str(e)}\n{traceback.format_exc()}")
            # Return dataframe with no exits if there's an error
            dataframe['exit_long'] = 0
            dataframe['exit_short'] = 0
            dataframe['exit_reason'] = ''  # Ensure the column exists even if there's an error
            return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                         current_rate: float, current_profit: float, after_fill: bool = False,
                         **kwargs) -> Optional[float]:
        """
        Custom stop loss based on ATR and support/resistance levels
        """
        try:
            # Enhanced logging
            # print(f"Custom stoploss called for {pair}: profit={current_profit:.2%}, after_fill={after_fill}")

            # Get the dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 1:
                return None

            # Get the last candle
            last_candle = dataframe.iloc[-1].squeeze()

            # For long positions
            if not trade.is_short:
                # Use ATR-based stop loss (2 * ATR)
                atr_stop = last_candle['low'] - (self.trailing_stop_ratio.value * last_candle['atr'])

                # Use the more conservative stop (higher for long)
                stop_loss_price = max(
                    atr_stop,
                    last_candle.get(f'trough_{self.primary_timeframe}', 0) * 0.998 if f'trough_{self.primary_timeframe}' in last_candle and not pd.isna(last_candle[f'trough_{self.primary_timeframe}']) else atr_stop
                )

                # Ensure stop is not too tight (at least 0.01% below entry)
                min_stop = trade.open_rate * 0.999
                stop_loss_price = max(stop_loss_price, min_stop)

            # For short positions
            else:
                # Use ATR-based stop loss (2 * ATR)
                atr_stop = last_candle['high'] + (self.trailing_stop_ratio.value * last_candle['atr'])

                # Use the more conservative stop (lower for short)
                stop_loss_price = min(
                    atr_stop,
                    last_candle.get(f'peak_{self.primary_timeframe}', float('inf')) * 1.002 if f'peak_{self.primary_timeframe}' in last_candle and not pd.isna(last_candle[f'peak_{self.primary_timeframe}']) else atr_stop
                )

                # Ensure stop is not too tight (at least 0.1% above entry)
                max_stop = trade.open_rate * 1.001
                stop_loss_price = min(stop_loss_price, max_stop)

            # Convert to percentage
            if stop_loss_price > 0:
                final_stoploss = stoploss_from_absolute(stop_loss_price, current_rate,
                                            is_short=trade.is_short, leverage=trade.leverage)

                # Only log when there's an actual change in stop loss value
                # Use a small epsilon for floating point comparison
                epsilon = 1e-8  # Small value to account for floating point precision
                current_stop_loss = trade.stop_loss if trade.stop_loss else 0

                # Check if the difference is significant (greater than epsilon)
                stop_loss_changed = abs(stop_loss_price - current_stop_loss) > epsilon

                if stop_loss_changed:
                    logger.info(
                        f"Stoploss update for {pair} "
                        f"({'short' if trade.is_short else 'long'}): "
                        f"price={stop_loss_price:.6f}, percent={final_stoploss:.4%}"
                    )

                return final_stoploss
            return None

        except Exception as e:
            logger.error(f"Error in custom_stop_loss: {str(e)}")
            return None

    def _get_collateral_per_trade_slot(self, total_equity: float) -> float:
        """
        Calculate collateral per trade slot
        based on total equity and available trade slots.
        Returns 0.0 if no slots are available or total_equity is 0.
        """
        if total_equity <= 1e-7:  # Effectively zero
            return 0.0

        open_trades_count = len(Trade.get_trades_proxy(is_open=True))
        # max_open_trades from strategy config
        max_open_trades = self.config.get('max_open_trades', 1)
        if not isinstance(max_open_trades, int) or max_open_trades <= 0:
            logger.warning(f"Invalid max_open_trades value: {max_open_trades}. Defaulting to 1.")
            max_open_trades = 1

        if open_trades_count >= max_open_trades:
            return 0.0  # No slots available

        available_slots = max_open_trades - open_trades_count
        # This check should ideally not be needed if open_trades_count < max_open_trades
        # but as a safeguard:
        if available_slots <= 0:
            return 0.0

        collateral_per_slot = total_equity / available_slots
        logger.debug(f"collateral per slot: {collateral_per_slot} {total_equity} {available_slots}")
        return collateral_per_slot

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                        proposed_stake: float, min_stake: Optional[float], max_stake: float,
                        leverage: float, entry_tag: Optional[str], side: str,
                        **kwargs) -> float:

        self._force_leverage_one_for_this_trade = False # Reset at the beginning
        total_equity = self.get_total_equity()
        collateral_per_slot = self._get_collateral_per_trade_slot(total_equity)

        # Your logic to determine ideal_stake, e.g., from proposed_stake or other calculations
        ideal_stake = proposed_stake # Placeholder for your actual logic

        actual_stake_to_use = ideal_stake

        if collateral_per_slot > 0 and collateral_per_slot < ideal_stake:
            # Condition met: available collateral per slot is less than what we'd ideally stake.
            # So, we use this smaller collateral_per_slot as the stake.
            actual_stake_to_use = collateral_per_slot
            # And signal the leverage() method to use leverage 1.0 for this trade.
            self._force_leverage_one_for_this_trade = True

        # Ensure stake is within min/max limits
        if min_stake is not None:
            actual_stake_to_use = max(actual_stake_to_use, min_stake)
        actual_stake_to_use = min(actual_stake_to_use, max_stake)

        logger.debug(f"Actual_stake_to_use ({pair}): {actual_stake_to_use} {collateral_per_slot} {ideal_stake}")
        return actual_stake_to_use

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Calculate leverage based on maximum risk per trade.
        The goal is to size the position such that if the stop-loss (trough_15m or peak_15m)
        is hit, the loss is no more than max_risk_per_trade of total equity.

        - Sets the maximum risk as a modifiable constant (max_risk_per_trade).
        - Risk in stake currency is (total_equity * max_risk_per_trade).
        - Desired position size (base currency) = risk_amount / (current_rate - stop_loss_price).
        - Calculated leverage = (desired_position_size * current_rate) / stake_for_this_trade.
        - If calculated leverage > max_leverage, do not enter (return 0.0).
        """
        # Check if custom_stake_amount decided to force leverage 1.0
        # This flag would be set by custom_stake_amount if it's active and makes such a decision.
        if hasattr(self, '_force_leverage_one_for_this_trade') and self._force_leverage_one_for_this_trade:
            self._force_leverage_one_for_this_trade = False  # Reset flag for the next trade
            return 1.0

        # Get total equity in stake currency
        total_equity = self.get_total_equity()
        logger.debug(f"Leverage: Calculating total equity for {pair}: {total_equity}")

        if total_equity <= 1e-7: # Effectively zero equity
            return 0.0 # Not enough equity to calculate leverage

        # Calculate risk amount in stake currency
        risk_amount_stake_curr = (total_equity * self.max_risk_per_trade.value)

        analyzed_df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if analyzed_df.empty:
            logger.warning(f"Leverage: Empty dataframe for pair {pair}, cannot determine stop-loss.")
            return 0.0 # Cannot determine stop loss, do not trade
        last_candle = analyzed_df.iloc[-1].squeeze()

        stop_loss_price = None
        price_diff_to_stop = 0.0

        if side == 'long':
            raw_stop_price = last_candle.get(f'trough_{self.primary_timeframe}')  # From primary informative
            if pd.isna(raw_stop_price):
                logger.warning(f"Leverage: trough_{self.primary_timeframe} is NaN for {pair} on {current_time}.")
                return 0.0  # Stop-loss level not found or NaN
            stop_loss_price = raw_stop_price * 0.998
            if current_rate <= stop_loss_price:
                return 0.0  # Invalid stop-loss for long
            price_diff_to_stop = current_rate - stop_loss_price
        elif side == 'short':
            raw_stop_price = last_candle.get(f'peak_{self.primary_timeframe}')  # From primary informative
            if pd.isna(raw_stop_price):
                logger.warning(f"Leverage: peak_{self.primary_timeframe} is NaN for {pair} on {current_time}.")
                return 0.0  # Stop-loss level not found or NaN
            stop_loss_price = raw_stop_price * 1.002
            if current_rate >= stop_loss_price:
                return 0.0  # Invalid stop-loss for short
            price_diff_to_stop = stop_loss_price - current_rate
        else:
            logger.error(f"Leverage: Invalid side '{side}' received.")
            return 0.0 # Should not happen

        if price_diff_to_stop <= 1e-7: # Avoid division by zero or very small stop distance
            return 0.0 # Stop too close, do not enter

        # Desired position size in base currency
        desired_position_size_base = risk_amount_stake_curr / price_diff_to_stop
        # Desired position value in stake currency
        desired_position_value_stake_curr = desired_position_size_base * current_rate

        # Collateral Freqtrade would allocate for this trade slot by default.
        collateral_for_this_trade_slot = self._get_collateral_per_trade_slot(total_equity)

        if collateral_for_this_trade_slot <= 1e-7: # Effectively zero collateral per slot
            return 0.0 # No collateral available per slot, do not trade

        required_leverage = desired_position_value_stake_curr / collateral_for_this_trade_slot

        if required_leverage <= 1e-7: # Effectively zero or negative desired leverage
            return 0.0 # Do not trade

        if required_leverage > max_leverage:
            return 0.0  # Required leverage too high, do not enter
        if required_leverage < 1.0:
            final_leverage = 1.0  # Use at least 1x leverage if conditions allow a trade
        else:
            final_leverage = required_leverage

        # Ensure leverage is capped by max_leverage
        final_leverage = min(final_leverage, max_leverage)

        return float(round(final_leverage, 4)) # Round to a sensible precision

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Called right after an order fills.
        """
        logger.info(f"Order filled callback triggered for {pair}: order_side={order.ft_order_side}, order_type={order.order_type}")

        # Exit if order is not an entry order
        if order.ft_order_side != trade.entry_side:
            # logger.info(f"Skipping non-entry order: {order.ft_order_side}")
            return None

        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        stop_loss_price = None
        price_diff_to_stop = 0.0

        side = 'long' if not trade.is_short else 'short'

        if side == 'long':
            raw_stop_price = last_candle.get(f'trough_{self.primary_timeframe}')  # From primary informative
            stop_loss_price = raw_stop_price * 0.998
            price_diff_to_stop = trade.open_rate - stop_loss_price
            take_profit_price = trade.open_rate + price_diff_to_stop
        elif side == 'short':
            raw_stop_price = last_candle.get(f'peak_{self.primary_timeframe}')  # From primary informative
            stop_loss_price = raw_stop_price * 1.002
            price_diff_to_stop = stop_loss_price - trade.open_rate
            take_profit_price = trade.open_rate - price_diff_to_stop
        else:
            logger.error(f"Order Filled: Invalid side '{side}' received.")
            return None # Should not happen

        # Log the take profit price being set
        logger.info(f"Setting take_profit_price={take_profit_price} for {pair}")
        trade.set_custom_data(key='take_profit_price', value=take_profit_price)

        return None


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        """
        Adjust trade position based on take profit conditions.

        When the price reaches the take profit level for the first time,
        reduce the position by 50% to lock in some profits while letting
        the remaining position continue to run.

        This is only done once per trade to avoid multiple reductions.

        IMPORTANT: The return value represents stake currency amount to reduce,
        NOT a percentage. To reduce by 50%, we must return -0.5 * trade.stake_amount.
        """

        if trade.has_open_orders:
            # Only act if no orders are open
            return

        take_profit_price = trade.get_custom_data(key='take_profit_price')
        take_profit_reduced = trade.get_custom_data(key='take_profit_reduced', default=False)

        # Check if we've reached take profit price and haven't reduced position yet
        # For long positions: current_rate >= take_profit_price
        # For short positions: current_rate <= take_profit_price
        take_profit_reached = False
        if take_profit_price is not None and not take_profit_reduced:
            if not trade.is_short:  # Long position
                take_profit_reached = current_rate >= take_profit_price
            else:  # Short position
                take_profit_reached = current_rate <= take_profit_price

        if take_profit_reached:
            # Mark that we've reduced the position at take profit
            trade.set_custom_data(key='take_profit_reduced', value=True)

            side_text = "short" if trade.is_short else "long"
            logger.info(f"Take profit reached for {trade.pair} ({side_text}) at {current_rate:.6f} "
                       f"(target: {take_profit_price:.6f}). Reducing position by 50%.")

            # Calculate the correct stake amount to reduce position by exactly 50%
            # FreqTrade formula: amount_to_exit = abs(stake_amount) * trade.amount / trade.stake_amount
            # To exit 50% of position: 0.5 * trade.amount = abs(stake_amount) * trade.amount / trade.stake_amount
            # Solving: stake_amount = -0.5 * trade.stake_amount (negative for reduction)
            reduction_stake_amount = -0.5 * trade.stake_amount

            # Calculate expected amount to be exited for validation
            expected_exit_amount = abs(reduction_stake_amount) * trade.amount / trade.stake_amount
            expected_exit_percentage = (expected_exit_amount / trade.amount) * 100

            logger.debug(f"Position reduction calculation for {trade.pair}:")
            logger.debug(f"  Current position: {trade.amount:.8f} {trade.base_currency}")
            logger.debug(f"  Current stake: {trade.stake_amount:.6f} {trade.stake_currency}")
            logger.debug(f"  Reduction stake amount: {reduction_stake_amount:.6f}")
            logger.debug(f"  Expected exit amount: {expected_exit_amount:.8f} ({expected_exit_percentage:.1f}%)")

            return reduction_stake_amount

        # If we've already reduced at take profit, let the remaining position run
        return None

    def plot_annotations(
        self, pair: str, start_date: datetime, end_date: datetime, dataframe: DataFrame, **kwargs
    ) -> list[AnnotationType]:
        """
        Retrieve area annotations for a chart.
        Creates area annotations between primary peaks and primary troughs to highlight
        periods of significant price movements.

        :param pair: Pair that's currently analyzed
        :param start_date: Start date of the chart data being requested
        :param end_date: End date of the chart data being requested
        :param dataframe: DataFrame with the analyzed data for the chart
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: List of AnnotationType objects
        """
        annotations = []

        # Check if we have the required columns
        peak_col = f'peak_{self.primary_timeframe}'
        trough_col = f'trough_{self.primary_timeframe}'

        if peak_col not in dataframe.columns or trough_col not in dataframe.columns:
            logger.warning(f"Peak/trough columns not found for {pair}. Available columns: {dataframe.columns.tolist()}")
            return annotations

        # Filter dataframe to the requested date range
        df_filtered = dataframe[
            (dataframe['date'] >= start_date) &
            (dataframe['date'] <= end_date)
        ].copy()

        if df_filtered.empty:
            return annotations

        # Identify significant peak and trough changes
        df_filtered['peak_change'] = df_filtered[peak_col] != df_filtered[peak_col].shift(1)
        df_filtered['trough_change'] = df_filtered[trough_col] != df_filtered[trough_col].shift(1)
        df_filtered['significant_change'] = df_filtered['peak_change'] | df_filtered['trough_change']

        # Get transition points where peaks or troughs change
        transition_points = df_filtered[df_filtered['significant_change']].copy()

        if len(transition_points) < 2:
            return annotations

        # Create ranges between transition points
        ranges = []
        for i in range(len(transition_points) - 1):
            current_point = transition_points.iloc[i]
            next_point = transition_points.iloc[i + 1]

            # Determine the relationship type for this range
            current_peak = current_point[peak_col]
            current_trough = current_point[trough_col]
            next_peak = next_point[peak_col]
            next_trough = next_point[trough_col]

            # Classify the range based on what changed
            range_type = None
            if current_peak != next_peak and current_trough != next_trough:
                # Both changed - determine dominant movement
                peak_change_pct = abs(next_peak - current_peak) / current_peak if current_peak > 0 else 0
                trough_change_pct = abs(next_trough - current_trough) / current_trough if current_trough > 0 else 0
                if peak_change_pct > trough_change_pct:
                    range_type = "peak_to_peak"
                else:
                    range_type = "trough_to_trough"
            elif current_peak != next_peak:
                range_type = "peak_to_peak"
            elif current_trough != next_trough:
                range_type = "trough_to_trough"
            else:
                continue  # No significant change

            ranges.append({
                'start': current_point['date'],
                'end': next_point['date'],
                'type': range_type,
                'start_peak': current_peak,
                'start_trough': current_trough,
                'end_peak': next_peak,
                'end_trough': next_trough
            })

        # Merge adjacent ranges of the same type to reduce annotation count
        merged_ranges = []
        if ranges:
            current_range = ranges[0]

            for next_range in ranges[1:]:
                # Check if ranges are adjacent and of the same type
                if (current_range['type'] == next_range['type'] and
                    current_range['end'] == next_range['start']):
                    # Merge ranges
                    current_range['end'] = next_range['end']
                    current_range['end_peak'] = next_range['end_peak']
                    current_range['end_trough'] = next_range['end_trough']
                else:
                    # Add current range and start new one
                    merged_ranges.append(current_range)
                    current_range = next_range

            # Add the last range
            merged_ranges.append(current_range)

        # Create annotations from merged ranges
        for range_data in merged_ranges:
            # Choose color based on range type - soft, semi-transparent colors for dark background
            if range_data['type'] == "peak_to_peak":
                color = "rgba(255, 182, 193, 0.3)"  # Light pink - for peak transitions
                label = "Peak Transition"
            else:  # trough_to_trough
                color = "rgba(173, 216, 230, 0.3)"  # Light blue - for trough transitions
                label = "Trough Transition"

            annotations.append({
                "type": "area",
                "label": label,
                "start": range_data['start'],
                "end": range_data['end'],
                # Omitting y_start and y_end will result in a vertical area spanning the whole height of the chart
                "color": color,
            })

        logger.debug(f"Created {len(annotations)} peak-trough annotations for {pair}")
        return annotations