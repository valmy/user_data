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
import traceback  # Import traceback for detailed error logging

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
    primary_timeframe = "15m"  # This can remain for your internal logic if needed
    major_timeframe = "1h"
    long_timeframe = "4h"

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
    use_custom_stoploss = True

    signal_timeframe_minutes = timeframe_to_minutes(timeframe)
    primary_timeframe_minutes = timeframe_to_minutes(primary_timeframe)
    major_timeframe_minutes = timeframe_to_minutes(major_timeframe)
    long_timeframe_minutes = timeframe_to_minutes(long_timeframe)

    # Calculate ratios
    if signal_timeframe_minutes == 0:
        ratio_primary_to_signal = float("inf")  # Or handle as an error
    else:
        ratio_primary_to_signal = primary_timeframe_minutes / signal_timeframe_minutes

    if primary_timeframe_minutes == 0:
        ratio_major_to_primary = float("inf")  # Or handle as an error
    else:
        ratio_major_to_primary = major_timeframe_minutes / primary_timeframe_minutes

    # ratio major to signal
    ratio_major_to_signal = major_timeframe_minutes / signal_timeframe_minutes

    if long_timeframe_minutes == 0:
        ratio_long_to_signal = float("inf")  # Or handle as an error
    else:
        ratio_long_to_signal = long_timeframe_minutes / signal_timeframe_minutes

    # Number of candles the strategy requires before producing valid signals
    # Ensure it's an integer using int() and max() to prevent float values
    startup_candle_count: int = int(max(50, 3 * ratio_major_to_signal))

    # Trigger type
    use_lrsi_trigger = BooleanParameter(default=True, space="buy", optimize=False)
    # Parameters for cradle convergence
    use_cradle_trigger = BooleanParameter(default=True, space="buy", optimize=False)
    convergence_window = IntParameter(3, 10, default=5, space="buy", optimize=False)
    use_breakout_trigger = BooleanParameter(default=False, space="buy", optimize=False)

    # Parameters for tuning
    volume_threshold = DecimalParameter(
        1.0, 4.0, default=2, decimals=1, space="buy", optimize=True
    )

    # Laguerre RSI parameters
    laguerre_gamma = DecimalParameter(
        0.6, 0.8, default=0.68, decimals=2, space="buy", load=True, optimize=False
    )
    small_candle_ratio = DecimalParameter(
        1.0, 5.0, default=2.0, decimals=1, space="buy", load=True, optimize=True
    )
    buy_laguerre_level = DecimalParameter(
        0.1, 0.4, default=0.2, decimals=1, space="buy", load=True, optimize=False
    )
    sell_laguerre_level = DecimalParameter(
        0.6, 0.9, default=0.8, decimals=1, space="sell", load=True, optimize=False
    )  # For short entry, cross below this

    # Choppiness Index parameters
    primary_chop_threshold = IntParameter(
        35, 60, default=45, space="buy", load=True, optimize=True
    )
    major_chop_threshold = IntParameter(
        35, 50, default=40, space="buy", load=False, optimize=False
    )

    rr_ratio = DecimalParameter(
        1.0, 5.0, default=2.0, decimals=1, space="buy", load=True, optimize=True
    )

    # Custom trade size parameters
    max_risk_per_trade = DecimalParameter(
        0.01, 0.05, default=0.02, decimals=3, space="buy", load=True, optimize=False
    )

    # Sell parameters
    trailing_stop_ratio = DecimalParameter(
        0.05, 0.5, default=0.2, decimals=2, space="sell", load=True, optimize=True
    )

    atr_stop_ratio = DecimalParameter(
        0.05, 10.0, default=5.0, decimals=1, space="sell", load=True, optimize=True
    )

    use_take_profit_2 = BooleanParameter(
        default=True, space="sell", optimize=True
    )

    def is_hyperopt_mode(self) -> bool:
        """Check if the current run mode is hyperopt"""
        return self.dp.runmode.value == "hyperopt"

    def get_total_equity(self):
        if self.is_hyperopt_mode():
            # Get values from config, with defaults if not set
            ratio = self.config.get("tradable_balance_ratio", 1.0)
            wallet = self.config.get("dry_run_wallet", 1000)
            logger.debug(
                f"get_total_equity: Using config values. Ratio: {ratio}, Wallet: {wallet}"
            )
            return ratio * wallet
        else:
            logger.debug(
                f"get_total_equity: Using live wallet balance: {self.wallets.get_total_stake_amount()}"
            )
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

        # Long timeframe for trend confirmation
        for pair in pairs:
            informative_pairs.append((pair, self.long_timeframe))

        return informative_pairs


    @informative(primary_timeframe)
    def populate_informative_primary(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        Populate indicators for primary trend identification on primary_timeframe timeframe
        """

        # Define confirmation ema
        dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)

        # Donchian Channels (using 5-period window)
        # These are used for trend identification
        dataframe["donchian_upper"] = dataframe["high"].rolling(window=5).max()
        dataframe["donchian_lower"] = dataframe["low"].rolling(window=5).min()

        # Identify peaks: where donchian_upper equals the high from 3 periods ago
        dataframe["peak"] = np.where(
            dataframe["donchian_upper"] == dataframe["high"].shift(2),
            dataframe["donchian_upper"],
            np.nan,
        )
        dataframe["peak"] = dataframe["peak"].ffill()

        # Identify troughs: where donchian_lower equals the low from 3 periods ago
        dataframe["trough"] = np.where(
            dataframe["donchian_lower"] == dataframe["low"].shift(2),
            dataframe["donchian_lower"],
            np.nan,
        )
        dataframe["trough"] = dataframe["trough"].ffill()

        # --- Trend detection for peak (for higher_high and lower_high) and trough (for higher_low and lower_low) ---
        # Initialize temporary columns for trend direction
        # 0: flat, 1: rising, -1: falling
        dataframe["peak_trend_temp"] = 0
        dataframe.loc[
            dataframe["high"] > dataframe["peak"].shift(1) * 1.001, "peak_trend_temp"
        ] = 1
        dataframe.loc[
            dataframe["peak"] > dataframe["peak"].shift(1) * 1.001, "peak_trend_temp"
        ] = 1
        dataframe.loc[
            dataframe["peak"] < dataframe["peak"].shift(1) * 0.999, "peak_trend_temp"
        ] = -1
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe["peak_trend_temp"] = (
            dataframe["peak_trend_temp"].replace(0, pd.NA).ffill()
        )
        # higher_high is True if the prevailing trend of donchian_upper is upwards (1)
        dataframe["higher_high"] = (
            (dataframe["peak_trend_temp"] == 1).fillna(False).astype(bool)
        )
        # lower_high is True if the prevailing trend of donchian_upper is downwards (-1)
        dataframe["lower_high"] = (
            (dataframe["peak_trend_temp"] == -1).fillna(False).astype(bool)
        )

        dataframe["trough_trend_temp"] = 0
        dataframe.loc[
            dataframe["trough"] > dataframe["trough"].shift(1) * 1.001,
            "trough_trend_temp",
        ] = 1
        dataframe.loc[
            dataframe["low"] < dataframe["trough"].shift(1) * 0.999, "trough_trend_temp"
        ] = -1
        dataframe.loc[
            dataframe["trough"] < dataframe["trough"].shift(1) * 0.999,
            "trough_trend_temp",
        ] = -1
        # 0: flat, 1: rising, -1: falling
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe["trough_trend_temp"] = (
            dataframe["trough_trend_temp"].replace(0, pd.NA).ffill()
        )

        # higher_low is True if the prevailing trend of donchian_lower is upwards (1)
        dataframe["higher_low"] = (
            (dataframe["trough_trend_temp"] == 1).fillna(False).astype(bool)
        )
        # lower_low is True if the prevailing trend of donchian_lower is downwards (-1)
        dataframe["lower_low"] = (
            (dataframe["trough_trend_temp"] == -1).fillna(False).astype(bool)
        )

        # Note: You might want to drop the temporary columns if they are not used elsewhere:
        dataframe.drop(["peak_trend_temp", "trough_trend_temp"], axis=1, inplace=True)

        # Choppiness Index
        dataframe["chop"] = pta.chop(
            dataframe["high"], dataframe["low"], dataframe["close"], length=14
        )
        return dataframe

    @informative(major_timeframe)
    def populate_informative_major(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        Populate indicators for major trend confirmation on major_timeframe timeframe
        """

        # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]
        dataframe["ha_high"] = heikinashi["high"]
        dataframe["ha_low"] = heikinashi["low"]
        dataframe["ha_bullish"] = (heikinashi["close"] > heikinashi["open"]).astype(
            bool
        )
        dataframe["ha_upswing"] = dataframe["ha_bullish"].rolling(4).sum() >= 3
        dataframe["ha_bearish"] = (heikinashi["close"] < heikinashi["open"]).astype(
            bool
        )
        dataframe["ha_downswing"] = dataframe["ha_bearish"].rolling(4).sum() >= 3

        # Donchian Channels (using 5-period window)
        # These are used for trend identification
        dataframe["donchian_upper"] = dataframe["high"].rolling(window=5).max()
        dataframe["donchian_lower"] = dataframe["low"].rolling(window=5).min()

        # Identify peaks: where donchian_upper equals the high from 3 periods ago
        dataframe["peak"] = np.where(
            dataframe["donchian_upper"] == dataframe["high"].shift(3),
            dataframe["donchian_upper"],
            np.nan,
        )
        dataframe["peak"] = dataframe["peak"].ffill()

        # Identify troughs: where donchian_lower equals the low from 3 periods ago
        dataframe["trough"] = np.where(
            dataframe["donchian_lower"] == dataframe["low"].shift(3),
            dataframe["donchian_lower"],
            np.nan,
        )
        dataframe["trough"] = dataframe["trough"].ffill()

        # --- Trend detection for peak (for higher_high and lower_high) and trough
        # (for higher_low and lower_low) ---
        # Initialize temporary columns for trend direction
        # 0: flat, 1: rising, -1: falling
        dataframe["peak_trend_temp"] = 0
        dataframe.loc[
            dataframe["high"] > dataframe["peak"].shift(1), "peak_trend_temp"
        ] = 1
        dataframe.loc[
            dataframe["peak"] < dataframe["peak"].shift(1), "peak_trend_temp"
        ] = -1
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe["peak_trend_temp"] = (
            dataframe["peak_trend_temp"].replace(0, pd.NA).ffill()
        )
        # higher_high is True if the prevailing trend of donchian_upper is upwards (1)
        dataframe["higher_high"] = (
            (dataframe["peak_trend_temp"] == 1).fillna(False).astype(bool)
        )
        # lower_high is True if the prevailing trend of donchian_upper is downwards (-1)
        dataframe["lower_high"] = (
            (dataframe["peak_trend_temp"] == -1).fillna(False).astype(bool)
        )

        dataframe["trough_trend_temp"] = 0
        dataframe.loc[
            dataframe["trough"] > dataframe["trough"].shift(1), "trough_trend_temp"
        ] = 1
        dataframe.loc[
            dataframe["low"] < dataframe["trough"].shift(1), "trough_trend_temp"
        ] = -1
        # 0: flat, 1: rising, -1: falling
        # Replace 0s (flat periods) with NA, then forward-fill the last known trend
        dataframe["trough_trend_temp"] = (
            dataframe["trough_trend_temp"].replace(0, pd.NA).ffill()
        )

        # higher_low is True if the prevailing trend of donchian_lower is upwards (1)
        dataframe["higher_low"] = (
            (dataframe["trough_trend_temp"] == 1).fillna(False).astype(bool)
        )
        # lower_low is True if the prevailing trend of donchian_lower is downwards (-1)
        dataframe["lower_low"] = (
            (dataframe["trough_trend_temp"] == -1).fillna(False).astype(bool)
        )

        # Note: You might want to drop the temporary columns if they are not used elsewhere:
        dataframe.drop(["peak_trend_temp", "trough_trend_temp"], axis=1, inplace=True)

        # Choppiness Index
        dataframe["chop"] = pta.chop(
            dataframe["high"], dataframe["low"], dataframe["close"], length=14
        )

        return dataframe

    @informative(long_timeframe)
    def populate_informative_long(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        Populate indicators for long trend confirmation on long_timeframe timeframe
        """

        # Choppiness Index
        dataframe["chop"] = pta.chop(
            dataframe["high"], dataframe["low"], dataframe["close"], length=14
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds indicators for secondary trends and generates buy/sell signals
        """
        # Secondary trend indicators
        dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # Define in_cradle zone when the current candle is within the cradle zone,
        # which is defined as the range between ema10 and ema20
        ema_min = dataframe[["ema10", "ema20"]].min(axis=1)
        ema_max = dataframe[["ema10", "ema20"]].max(axis=1)
        dataframe["in_cradle"] = (
            (dataframe["high"] >= ema_min) & (dataframe["low"] <= ema_max)
        )

        # Laguerre RSI
        dataframe["laguerre"] = indicators.laguerre(
            dataframe, gamma=self.laguerre_gamma.value
        )

        # Momentum and volume indicators
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Volume confirmation
        dataframe["volume_mean"] = dataframe["volume"].rolling(10).mean()
        dataframe["volume_increased"] = dataframe["volume"] > (
            dataframe["volume_mean"] * self.volume_threshold.value
        )

        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["bullish_candle"] = (
            dataframe["close"] > dataframe["low"] + 0.7 * dataframe["candle_range"]
        )
        dataframe["bearish_candle"] = (
            dataframe["close"] < dataframe["high"] - 0.7 * dataframe["candle_range"]
        )
        # Small candle condition: candle range must be smaller than small_candle_ratio * ATR
        dataframe["small_candle"] = dataframe["candle_range"] < (
            self.small_candle_ratio.value * dataframe["atr"]
        )

        # MACD
        dataframe.ta.macd(fast=12, slow=26, signal=9, append=True)

        # Donchian Channels (using 36-period window)
        major_period = round(3 * self.ratio_major_to_signal)
        primary_period = round(3 * self.ratio_primary_to_signal)

        # Calculate rolling windows with integer periods
        dataframe["donchian_upper"] = (
            dataframe["high"].rolling(window=major_period, min_periods=1).max()
        )
        dataframe["donchian_lower"] = (
            dataframe["low"].rolling(window=major_period, min_periods=1).min()
        )

        dataframe["stop_upper"] = dataframe["high"].rolling(window=primary_period, min_periods=1).max()
        dataframe["stop_lower"] = dataframe["low"].rolling(window=primary_period, min_periods=1).min()

        # long_target as the higher between donchian_upper and peak in major timeframe
        dataframe["long_target"] = dataframe[["donchian_upper", f"peak_{self.major_timeframe}"]].max(axis=1)
        # short_target as the lower between donchian_lower and trough in major timeframe
        dataframe["short_target"] = dataframe[["donchian_lower", f"trough_{self.major_timeframe}"]].min(axis=1)

        # long stop as the lower between stop_lower, trough in primary timeframe,
        # and close - 2 * atr
        dataframe["close_minus_2atr"] = dataframe["close"] - 2 * dataframe["atr"]
        dataframe["long_stop"] = dataframe[["stop_lower", f"trough_{self.primary_timeframe}", "close_minus_2atr"]].min(axis=1)
        # short stop as the higher between stop_upper, peak in primary timeframe,
        dataframe["close_plus_2atr"] = dataframe["close"] + 2 * dataframe["atr"]
        dataframe["short_stop"] = dataframe[["stop_upper", f"peak_{self.primary_timeframe}", "close_plus_2atr"]].max(axis=1)

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
            df["enter_long"] = 0
            df["enter_short"] = 0
            df["enter_tag"] = ""

            # Calculate conditions with error handling
            try:
                # --- Debugging for entry condition error ---
                ptf_chop_col = f"chop_{self.primary_timeframe}"
                mtf_chop_col = f"chop_{self.major_timeframe}"

                if ptf_chop_col not in df.columns or mtf_chop_col not in df.columns:
                    logger.error(
                        f"Chop columns missing! Primary: {ptf_chop_col in df.columns}, Major: {mtf_chop_col in df.columns}. All columns: {df.columns.to_list()}"
                    )
                    df["enter_long"] = 0
                    df["enter_short"] = 0
                    return df

                ptf_thresh_val = self.primary_chop_threshold.value
                mtf_thresh_val = self.major_chop_threshold.value

                cond_ptf_chop = df[ptf_chop_col] > ptf_thresh_val
                cond_mtf_chop = df[mtf_chop_col] > mtf_thresh_val

                # --- Debugging ---
                # Get the ha_upswing from the informative major timeframe
                ha_upswing_col = f"ha_upswing_{self.major_timeframe}"
                if ha_upswing_col not in df.columns:
                    logger.error(
                        f"ha_upswing column {ha_upswing_col} not found in dataframe columns: {df.columns.to_list()}"
                    )
                    return df

                # Get the ha_downswing from the informative major timeframe
                ha_downswing_col = f"ha_downswing_{self.major_timeframe}"
                if ha_downswing_col not in df.columns:
                    logger.error(
                        f"ha_downswing column {ha_downswing_col} not found in dataframe columns: {df.columns.to_list()}"
                    )
                    return df

                # --- End Debugging ---

                # Pre-calculate common conditions for better performance
                df["strong_volume"] = df["volume"] > (df["volume_mean"] * 1.5)
                df["bullish_candle"] = df["close"] > df["open"]
                df["bearish_candle"] = df["close"] < df["open"]
                back_range = int(3 * self.ratio_primary_to_signal)
                df["above_resistance"] = (
                    df["low"].rolling(window=back_range).min()
                    >= df[f"trough_{self.primary_timeframe}"]
                )
                df["below_support"] = (
                    df["high"].rolling(window=back_range).max()
                    <= df[f"peak_{self.primary_timeframe}"]
                )

                # Add close - 2 * atr requirement for long entries
                df["close_minus_2atr"] = df["close"] - 2 * df["atr"]
                df["close_above_close_minus_2atr"] = df["close"] > df["close_minus_2atr"]

                # RR ratio calculation
                df["long_rr_ratio"] = (
                    (df[f"donchian_upper_{self.major_timeframe}"] - df["close"])
                    / (df["close"] - df[f"trough_{self.primary_timeframe}"])
                )
                df["short_rr_ratio"] = (
                    (df["close"] - df[f"donchian_lower_{self.major_timeframe}"])
                    / (df[f"peak_{self.primary_timeframe}"] - df["close"])
                )

                df["long_cradle_rr_ratio"] = (
                    (df[f"donchian_upper_{self.major_timeframe}"] - df["high"])
                    / (df["high"] - df["stop_lower"])
                )
                df["short_cradle_rr_ratio"] = (
                    (df["low"] - df[f"donchian_lower_{self.major_timeframe}"])
                    / (df["stop_upper"] - df["low"])
                )

                # --- Trigger conditions ---
                # LRSI Triggers
                long_lrsi_trigger = qtpylib.crossed_above(df["laguerre"], self.buy_laguerre_level.value)
                short_lrsi_trigger = qtpylib.crossed_below(df["laguerre"], self.sell_laguerre_level.value)

                # Detect peaks and troughs in signal timeframe (5m)
                # Peak: high is higher than previous and next candles
                df["is_peak"] = (
                    (df["high"] > df["high"].shift(1)) &
                    (df["high"] > df["high"].shift(2)) &
                    (df["high"] >= df["high"].shift(-1)) &
                    (df["high"] >= df["high"].shift(-2))
                )

                # Trough: low is lower than previous and next candles
                df["is_trough"] = (
                    (df["low"] < df["low"].shift(1)) &
                    (df["low"] < df["low"].shift(2)) &
                    (df["low"] <= df["low"].shift(-1)) &
                    (df["low"] <= df["low"].shift(-2))
                )

                # Identify peak and trough values
                df["peak_value"] = np.where(df["is_peak"], df["high"], np.nan)
                df["peak_value"] = df["peak_value"].ffill()
                df["trough_value"] = np.where(df["is_trough"], df["low"], np.nan)
                df["trough_value"] = df["trough_value"].ffill()

                # Identify increasing/decreasing peaks and troughs
                # For peaks: compare current peak with previous peak
                window = self.convergence_window.value
                df["peak_increasing"] = df["peak_value"] > df["peak_value"].shift(window)
                df["peak_decreasing"] = df["peak_value"] < df["peak_value"].shift(window)

                # For troughs: compare current trough with previous trough
                df["trough_increasing"] = df["trough_value"] > df["trough_value"].shift(window)
                df["trough_decreasing"] = df["trough_value"] < df["trough_value"].shift(window)

                # Identify MACD peaks and troughs
                df["macd_peak"] = (
                    (df["MACD_12_26_9"] > df["MACD_12_26_9"].shift(1)) &
                    (df["MACD_12_26_9"] > df["MACD_12_26_9"].shift(2)) &
                    (df["MACD_12_26_9"] >= df["MACD_12_26_9"].shift(-1)) &
                    (df["MACD_12_26_9"] >= df["MACD_12_26_9"].shift(-2))
                )

                df["macd_trough"] = (
                    (df["MACD_12_26_9"] < df["MACD_12_26_9"].shift(1)) &
                    (df["MACD_12_26_9"] < df["MACD_12_26_9"].shift(2)) &
                    (df["MACD_12_26_9"] <= df["MACD_12_26_9"].shift(-1)) &
                    (df["MACD_12_26_9"] <= df["MACD_12_26_9"].shift(-2))
                )

                # Identify MACD peak and trough values
                df["macd_peak_value"] = np.where(df["macd_peak"], df["MACD_12_26_9"], np.nan)
                df["macd_peak_value"] = df["macd_peak_value"].ffill()
                df["macd_trough_value"] = np.where(df["macd_trough"], df["MACD_12_26_9"], np.nan)
                df["macd_trough_value"] = df["macd_trough_value"].ffill()

                # Identify increasing/decreasing MACD peaks and troughs
                # For MACD peaks: compare current peak with previous peak
                df["macd_peak_increasing"] = df["macd_peak_value"] > df["macd_peak_value"].shift(window)
                df["macd_peak_decreasing"] = df["macd_peak_value"] < df["macd_peak_value"].shift(window)

                # For MACD troughs: compare current trough with previous trough
                df["macd_trough_increasing"] = df["macd_trough_value"] > df["macd_trough_value"].shift(window)
                df["macd_trough_decreasing"] = df["macd_trough_value"] < df["macd_trough_value"].shift(window)

                # Cradle Triggers
                long_cradle_base = (
                    df["in_cradle"]
                    & (df["ema20"] < df["ema10"])
                    & (df[f"higher_high_{self.primary_timeframe}"])
                    & (df[f"ema20_{self.primary_timeframe}"] < df[f"ema10_{self.primary_timeframe}"])
                    & (df["bullish_candle"])
                    & (df["small_candle"])
                )

                short_cradle_base = (
                    df["in_cradle"]
                    & (df["ema20"] > df["ema10"])
                    & (df[f"lower_low_{self.primary_timeframe}"])
                    & (df[f"ema20_{self.primary_timeframe}"] > df[f"ema10_{self.primary_timeframe}"])
                    & (df["bearish_candle"])
                    & (df["small_candle"])
                )

                # Additional condition: no candle in the previous window candles has highs below ema20 for long trades
                # and no candle in the previous window candles has lows above ema20 for short trades
                window = self.convergence_window.value
                long_no_low_candles = True
                short_no_high_candles = True

                for i in range(1, window + 1):
                    long_no_low_candles = long_no_low_candles & (df["high"].shift(i) >= df["ema20"].shift(i))
                    short_no_high_candles = short_no_high_candles & (df["low"].shift(i) <= df["ema20"].shift(i))

                long_cradle_base = long_cradle_base & long_no_low_candles
                short_cradle_base = short_cradle_base & short_no_high_candles

                # Cradle Triggers with convergence confirmation
                long_cradle_trigger = long_cradle_base
                short_cradle_trigger = short_cradle_base

                # Apply convergence filter if enabled
                if self.use_cradle_trigger.value:
                    # For long entries, we want either:
                    # 1. Price peaks increasing and MACD peaks increasing (bullish convergence)
                    # 2. Price troughs increasing (bullish divergence)
                    long_cradle_trigger = long_cradle_base & (
                        df["peak_increasing"] & df["macd_peak_increasing"] & df["trough_increasing"]
                    )

                    # For short entries, we want either:
                    # 1. Price peaks decreasing and MACD peaks decreasing (bearish convergence)
                    # 2. Price troughs decreasing (bearish divergence)
                    short_cradle_trigger = short_cradle_base & (
                        df["peak_decreasing"] & df["macd_peak_decreasing"] & df["trough_decreasing"]
                    )

                # Breakout Triggers
                long_breakout_trigger = (
                    qtpylib.crossed_above(df["close"], df["donchian_upper"].shift(1))
                    & (df["bullish_candle"])
                )
                short_breakout_trigger = (
                    qtpylib.crossed_below(df["close"], df["donchian_lower"].shift(1))
                    & (df["bearish_candle"])
                )

                # --- Base Entry Conditions (excluding triggers) ---
                base_long_condition = (
                    df[ha_upswing_col] &
                    cond_mtf_chop &
                    df["strong_volume"] &
                    df["small_candle"]
                )
                base_short_condition = (
                    df[ha_downswing_col] &
                    cond_mtf_chop &
                    df["strong_volume"] &
                    df["small_candle"]
                )

                # --- Combine Triggers and Base Conditions ---
                # Long Entries
                if self.use_breakout_trigger.value:
                    long_condition = base_long_condition & long_breakout_trigger
                    df.loc[long_condition, ["enter_long", "enter_tag"]] = (1, "breakout")

                # Use cradle trigger
                if self.use_cradle_trigger.value:
                    long_rr_cond = df["long_cradle_rr_ratio"] >= 1.0
                    long_condition = base_long_condition & long_cradle_trigger & long_rr_cond
                    df.loc[long_condition, ["enter_long", "enter_tag"]] = (1, "cradle")

                if self.use_lrsi_trigger.value:
                    long_rr_cond = df["long_rr_ratio"] >= self.rr_ratio.value
                    long_condition = (base_long_condition
                        & cond_ptf_chop
                        & long_lrsi_trigger
                        & long_rr_cond)
                    df.loc[long_condition, ["enter_long", "enter_tag"]] = (1, "lrsi")

                # Short Entries
                if self.can_short:
                    if self.use_breakout_trigger.value:
                        short_condition = base_short_condition & short_breakout_trigger
                        df.loc[short_condition, ["enter_short", "enter_tag"]] = (1, "breakout")

                    # Use cradle trigger
                    if self.use_cradle_trigger.value:
                        short_rr_cond = df["short_cradle_rr_ratio"] >= 1.0
                        short_condition = base_short_condition & short_cradle_trigger & short_rr_cond
                        df.loc[short_condition, ["enter_short", "enter_tag"]] = (1, "cradle")

                    if self.use_lrsi_trigger.value:
                        short_rr_cond = df["short_rr_ratio"] >= self.rr_ratio.value
                        short_condition = (base_short_condition
                            & cond_ptf_chop
                            & short_lrsi_trigger
                            & short_rr_cond)
                        df.loc[short_condition, ["enter_short", "enter_tag"]] = (1, "lrsi")

                # Limit the number of signals to avoid over-trading
                # This part might need adjustment if multiple signals can be generated in one candle
                # For now, we assume the last trigger set wins, which is acceptable.

                return df

            except Exception as e:
                logger.error(
                    f"Error in entry conditions for {metadata['pair']}: {str(e)}\n{traceback.format_exc()}"
                )
                # Return dataframe with no signals if there's an error
                df["enter_long"] = 0
                df["enter_short"] = 0
                return df

        except Exception as e:
            logger.error(
                f"Critical error in populate_entry_trend for {metadata['pair']}: {str(e)}\n{traceback.format_exc()}"
            )
            # Return the original dataframe with no signals if something goes wrong
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on price conditions only
        """
        try:
            # Create a copy to avoid SettingWithCopyWarning
            df = dataframe.copy()

            # Initialize exit columns
            df["exit_long"] = 0
            df["exit_short"] = 0

            # Get entry tags from the dataframe if available (for backtesting)
            # In live trading, we'll need to get them from the trade object
            if "enter_tag" in df.columns:
                # For backtesting, we can use the enter_tag from the dataframe
                df["current_enter_tag"] = df["enter_tag"].ffill()
            else:
                # Default to lrsi if no enter_tag column
                df["current_enter_tag"] = "lrsi"

            try:
                # Define stop loss columns for different entry types
                long_stop_col = f"trough_{self.primary_timeframe}"
                short_stop_col = f"peak_{self.primary_timeframe}"

                # For cradle entries
                long_cradle_stop_col = "stop_lower"
                short_cradle_stop_col = "stop_upper"

                # For breakout entries
                long_breakout_stop_col = "close_minus_2atr"
                short_breakout_stop_col = "close_plus_2atr"

                missing_cols = []
                required_cols = [long_stop_col, short_stop_col, long_cradle_stop_col,
                                short_cradle_stop_col, long_breakout_stop_col, short_breakout_stop_col]

                for col in required_cols:
                    if col not in df.columns:
                        missing_cols.append(col)

                if missing_cols:
                    logger.error(
                        f"Exit condition columns missing: {missing_cols}. All columns: {df.columns.to_list()}"
                    )
                    return df  # Return df with no exits

                # Create conditions for different entry types
                is_lrsi_entry = df["current_enter_tag"] == "lrsi"
                is_cradle_entry = df["current_enter_tag"] == "cradle"
                is_breakout_entry = df["current_enter_tag"] == "breakout"

                # Exit LONG positions based on entry type
                exit_long_lrsi = (df["close"] < df[long_stop_col]) & is_lrsi_entry
                exit_long_cradle = (df["close"] < df[long_cradle_stop_col]) & is_cradle_entry
                exit_long_breakout = (df["close"] < df[long_breakout_stop_col]) & is_breakout_entry

                # Exit SHORT positions based on entry type
                exit_short_lrsi = (df["close"] > df[short_stop_col]) & is_lrsi_entry
                exit_short_cradle = (df["close"] > df[short_cradle_stop_col]) & is_cradle_entry
                exit_short_breakout = (df["close"] > df[short_breakout_stop_col]) & is_breakout_entry

                # Apply exit conditions
                df.loc[exit_long_lrsi | exit_long_cradle | exit_long_breakout, 'exit_long'] = 1
                df.loc[exit_short_lrsi | exit_short_cradle | exit_short_breakout, 'exit_short'] = 1

                return df
            except Exception as e_inner:
                logger.error(
                    f"Error in exit trend condition calculation for {metadata['pair']}: "
                    f"{str(e_inner)}\n{traceback.format_exc()}"
                )
                # Return dataframe with no exits if there's an error
                df['exit_long'] = 0
                df['exit_short'] = 0
                return df

        except Exception as e:
            logger.error(
                f"Critical error in populate_exit_trend for {metadata['pair']}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            # Return dataframe with no exits if there's an error
            dataframe["exit_long"] = 0
            dataframe["exit_short"] = 0
            return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Called right before placing a entry order.
        Timing: Called after populate_entry_trend and before the entry order is placed.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 2:
            return False

        last_candle = dataframe.iloc[-2]  # Previous candle

        if entry_tag == 'cradle':
            if side == "long":
                if rate > last_candle['high']:
                    return True
            elif side == "short":
                if rate < last_candle['low']:
                    return True
            return False

        elif entry_tag == 'breakout':
            if side == "long":
                if rate > last_candle['close_minus_2atr']:
                    return True
            elif side == "short":
                if rate < last_candle['close_plus_2atr']:
                    return True
            return False

        else: # default to "lrsi"
            if side == "long":
                if rate > last_candle["long_stop"]:
                    return True
            elif side == "short":
                if rate < last_candle["short_stop"]:
                    return True

            return False

        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        sell_reason: str,
        **kwargs,
    ) -> bool:
        """
        Called right before placing a regular sell order.
        Timing: Called after populate_exit_trend and before the sell order is placed.

        We use this to implement the same dynamic stop logic as a "soft stop" that
        only triggers on candle close, rather than the "hard stop" that custom_stoploss
        implements (which can be triggered by candle wicks).
        """
        try:
            # Only apply this logic if the sell reason is 'exit_signal' (from populate_exit_trend)
            if sell_reason != 'exit_signal':
                return True  # Allow other types of exits (stoploss, roi, etc.)

            # Get the dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 1:
                return True  # Allow exit if no data

            # Get the last candle
            last_candle = dataframe.iloc[-1].squeeze()

            # Get custom data from trade
            use_dynamic_stop = trade.get_custom_data(key="use_dynamic_stop", default=False)
            dynamic_stop = trade.get_custom_data(key="dynamic_stop", default=None)
            initial_stop_loss = trade.get_custom_data(key="initial_stop_loss", default=None)

            # For long positions
            if not trade.is_short:
                # If using dynamic stop, check against dynamic stop level
                if use_dynamic_stop and dynamic_stop is not None:
                    # Allow exit only if close is below dynamic stop
                    if last_candle["close"] >= dynamic_stop:
                        return False  # Prevent exit
                else:
                    # Check against initial stop loss (all follows the trough)
                    if last_candle["close"] >= initial_stop_loss:
                        return False  # Prevent exit

            # For short positions
            else:
                # If using dynamic stop, check against dynamic stop level
                if use_dynamic_stop and dynamic_stop is not None:
                    # Allow exit only if close is above dynamic stop
                    if last_candle["close"] <= dynamic_stop:
                        return False  # Prevent exit
                else:
                    # Check against initial stop loss (all follows the peak)
                    if last_candle["close"] <= initial_stop_loss:
                        return False  # Prevent exit

            # If we get here, allow the exit
            return True

        except Exception as e:
            logger.error(f"Error in confirm_trade_exit: {str(e)}")
            # In case of error, allow the exit to proceed
            return True

    def _set_trade_initial_values(self, trade: Trade, last_candle) -> float | None:
        """
        Calculate take profit levels based on entry type and trade side.
        Returns tuple of (take_profit_price, take_profit_2_price) or None if invalid.
        """
        stop_loss_price = None
        price_diff_to_stop = 0.0

        side = "long" if not trade.is_short else "short"
        entry_tag = trade.enter_tag if hasattr(trade, 'enter_tag') else 'lrsi'

        # Determine stop loss and take profit levels based on entry type
        if entry_tag == "cradle":
            if side == "long":
                raw_stop_price = last_candle.get("stop_lower")
                stop_loss_price = raw_stop_price * 0.995
                price_diff_to_stop = trade.open_rate - stop_loss_price
                take_profit_price = trade.open_rate + price_diff_to_stop
                take_profit_2_price = last_candle.get("long_target")
                if take_profit_2_price <= take_profit_price:
                    take_profit_2_price = trade.open_rate + 2 * price_diff_to_stop
            elif side == "short":
                raw_stop_price = last_candle.get("stop_upper")
                stop_loss_price = raw_stop_price * 1.005
                price_diff_to_stop = stop_loss_price - trade.open_rate
                take_profit_price = trade.open_rate - price_diff_to_stop
                take_profit_2_price = last_candle.get("short_target")
                if take_profit_2_price >= take_profit_price:
                    take_profit_2_price = trade.open_rate - 2 * price_diff_to_stop

        elif entry_tag == "breakout":
            if side == "long":
                raw_stop_price = last_candle.get("close_minus_2atr")
                stop_loss_price = raw_stop_price * 0.995
                price_diff_to_stop = trade.open_rate - stop_loss_price
                take_profit_price = trade.open_rate + price_diff_to_stop
                take_profit_2_price = last_candle.get("long_target")
                if take_profit_2_price <= take_profit_price:
                    take_profit_2_price = trade.open_rate + 2 * price_diff_to_stop
            elif side == "short":
                raw_stop_price = last_candle.get("close_plus_2atr")
                stop_loss_price = raw_stop_price * 1.005
                price_diff_to_stop = stop_loss_price - trade.open_rate
                take_profit_price = trade.open_rate - price_diff_to_stop
                take_profit_2_price = last_candle.get("short_target")
                if take_profit_2_price >= take_profit_price:
                    take_profit_2_price = trade.open_rate - 2 * price_diff_to_stop

        else:  # Default to lrsi
            if side == "long":
                raw_stop_price = last_candle.get("long_stop")
                stop_loss_price = raw_stop_price * 0.995
                price_diff_to_stop = trade.open_rate - stop_loss_price
                take_profit_price = trade.open_rate + price_diff_to_stop
                take_profit_2_price = last_candle.get("long_target")
                if take_profit_2_price <= take_profit_price:
                    take_profit_2_price = trade.open_rate + 2 * price_diff_to_stop
            elif side == "short":
                raw_stop_price = last_candle.get("short_stop")
                stop_loss_price = raw_stop_price * 1.005
                price_diff_to_stop = stop_loss_price - trade.open_rate
                take_profit_price = trade.open_rate - price_diff_to_stop
                take_profit_2_price = last_candle.get("short_target")
                if take_profit_2_price >= take_profit_price:
                    take_profit_2_price = trade.open_rate - 2 * price_diff_to_stop

        if side == "long":
            initial_trough = last_candle.get(f"trough_{self.primary_timeframe}")
            trade.set_custom_data(key="initial_trough", value=initial_trough)
        elif side == "short":
            initial_peak = last_candle.get(f"peak_{self.primary_timeframe}")
            trade.set_custom_data(key="initial_peak", value=initial_peak)
        else:
            logger.error(f"Order Filled: Invalid side '{side}' received.")
            return None  # Should not happen

        # Log the take profit price being set
        logger.info(
            f"Setting take_profit_price={take_profit_price} for {trade.pair}, "
            f"take_profit_2={take_profit_2_price}, "
            f"stop_loss={stop_loss_price:.6f}"
        )
        trade.set_custom_data(key="take_profit_price", value=take_profit_price)
        trade.set_custom_data(
            key="take_profit_2_price", value=take_profit_2_price
        )
        # Initialize dynamic stop with the initial stop loss for short positions
        trade.set_custom_data(key="initial_stop", value=stop_loss_price)
        trade.set_custom_data(key="dynamic_stop", value=stop_loss_price)

        return stop_loss_price

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> float | None:
        """
        Custom stop loss based on ATR and support/resistance levels, with dynamic trailing based on
        rising troughs for long positions and falling peaks for short positions
        """
        try:
            # Enhanced logging
            # print(f"{current_time.strftime('%Y-%m-%d %H:%M')} Custom stoploss called for {pair}: profit={current_profit:.2%}, after_fill={after_fill}")

            # Get the dataframe
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 1:
                return None

            # Get the last candle
            last_candle = dataframe.iloc[-1].squeeze()

            # Get the entry tag to determine stop loss method
            entry_tag = trade.enter_tag if hasattr(trade, 'enter_tag') else 'lrsi'

            take_profit_reduced = trade.get_custom_data(
                key="take_profit_reduced", default=False
            )

            # Get the dynamic stop and initial stop from trade custom data
            dynamic_stop = trade.get_custom_data(key="dynamic_stop", default=None)
            initial_stop = trade.get_custom_data(key="initial_stop", default=None)
            use_dynamic_stop = trade.get_custom_data(key="use_dynamic_stop", default=False)

            # Only initialize stop loss in custom_stoploss if it hasn't been set yet
            # (i.e., if order_filled didn't run or didn't set the values)
            if after_fill and not take_profit_reduced and initial_stop is None:
                stop_loss_price = self._set_trade_initial_values(
                    trade, last_candle
                )

            # If we're not after fill or the stop loss is already initialized, proceed with normal logic
            if not after_fill or initial_stop is not None:
                trailing_atr = self.atr_stop_ratio.value * last_candle['atr']

                # Determine if we should start using dynamic stop
                # Conditions to start using dynamic stop:
                # 1. Price has increased or decreased by 2x ATR
                # 2. Or the trough/peak has moved from its initial value
                if not use_dynamic_stop:
                    price_increased = (not trade.is_short and last_candle["high"] >= trade.open_rate + 2 * last_candle['atr'])
                    price_decreased = (trade.is_short and last_candle["low"] <= trade.open_rate - 2 * last_candle['atr'])
                    # time_since_entry = (current_time - trade.open_date).total_seconds() / 60  # minutes

                    # Determine if stop level has changed
                    if entry_tag == "cradle":
                        if not trade.is_short:
                            stop_changed = (last_candle.get("stop_lower", 0) * 0.995) != initial_stop
                        else:
                            stop_changed = (last_candle.get("stop_upper", float("inf")) * 1.005) != initial_stop
                    else: # breakout and lrsi
                        if not trade.is_short:
                            initial_trough = trade.get_custom_data(key="initial_trough")
                            stop_changed = (last_candle.get(f"trough_{self.primary_timeframe}", 0) != initial_trough)
                        else:
                            initial_peak = trade.get_custom_data(key="initial_peak")
                            stop_changed = (last_candle.get(f"peak_{self.primary_timeframe}", float("inf")) != initial_peak)

                    if (price_increased or price_decreased or  # profit
                        stop_changed):
                        # logger.debug(f"Enabling dynamic stop for {pair} ({'short' if trade.is_short else 'long'}): "
                        #       f"Profit: {current_profit:.2%}, "
                        #       f"Stop changed: {stop_changed}")
                        trade.set_custom_data(key="use_dynamic_stop", value=True)
                        use_dynamic_stop = True

                        # Debug logging for enabling dynamic stop
                        # logger.debug(
                        #     f"{current_time.strftime('%Y-%m-%d %H:%M')} Enabling dynamic stop for {pair} ({'short' if trade.is_short else 'long'}): "
                        #     f"Profit: {current_profit:.2%}, "
                        #     f"Stop changed: {stop_changed}"
                        #     f"Price increased: {price_increased}, Price decreased: {price_decreased}"
                        # )

                # For long positions
                if not trade.is_short:
                    # Get the current stop value based on entry type
                    if entry_tag == "cradle":
                        current_stop = last_candle.get("stop_lower", 0)
                    elif entry_tag == "breakout":
                        current_stop = last_candle.get("close_minus_2atr", 0)
                    else:  # Default to lrsi
                        current_stop = last_candle.get(f"trough_{self.primary_timeframe}", 0)

                    # Debug logging for current values
                    # logger.debug(
                    #     f"Long position {pair}: Current stop={current_stop:.6f}, "
                    #     f"Dynamic stop={dynamic_stop}, Initial stop={initial_stop}, "
                    #     f"Use dynamic={use_dynamic_stop}"
                    # )

                    # Update dynamic stop if we're using dynamic stop and current stop is higher
                    if use_dynamic_stop and dynamic_stop is not None and current_stop > dynamic_stop:
                        old_dynamic_stop = dynamic_stop
                        dynamic_stop = current_stop
                        trade.set_custom_data(key="dynamic_stop", value=dynamic_stop)

                        # Debug logging for dynamic stop update
                        # logger.debug(
                        #     f"{current_time.strftime('%Y-%m-%d %H:%M')} Updated dynamic stop for {pair} (long): "
                        #     f"{old_dynamic_stop:.6f} -> {dynamic_stop:.6f}"
                        # )

                    # Determine stop loss price based on conditions
                    if use_dynamic_stop and dynamic_stop is not None:
                        # Use dynamic stop with ATR buffer
                        atr_stop_price = last_candle["close"] - trailing_atr
                        stop_loss_price = max(dynamic_stop * 0.995, atr_stop_price)

                        # Debug logging for final stop calculation
                        # logger.debug(
                        #     f"Using dynamic stop for {pair} (long): "
                        #     f"Dynamic={dynamic_stop:.6f}, ATR stop={atr_stop_price:.6f}, "
                        #     f"Final={stop_loss_price:.6f}"
                        # )
                    else:
                        return None

                # For short positions
                else:
                    # Get the current stop value based on entry type
                    if entry_tag == "cradle":
                        current_stop = last_candle.get("stop_upper", float("inf"))
                    elif entry_tag == "breakout":
                        current_stop = last_candle.get("close_plus_2atr", float("inf"))
                    else:  # Default to lrsi
                        current_stop = last_candle.get(f"peak_{self.primary_timeframe}", float("inf"))

                    # Debug logging for current values
                    # logger.debug(
                    #     f"Short position {pair}: Current stop={current_stop:.6f}, "
                    #     f"Dynamic stop={dynamic_stop}, Initial stop={initial_stop}, "
                    #     f"Use dynamic={use_dynamic_stop}"
                    # )

                    # Update dynamic stop if we're using dynamic stop and current stop is lower
                    if use_dynamic_stop and dynamic_stop is not None and current_stop < dynamic_stop:
                        # old_dynamic_stop = dynamic_stop
                        dynamic_stop = current_stop
                        trade.set_custom_data(key="dynamic_stop", value=dynamic_stop)

                        # Debug logging for dynamic stop update
                        # logger.debug(
                        #     f"{current_time.strftime('%Y-%m-%d %H:%M')} Updated dynamic stop for {pair} (short): "
                        #     f"{old_dynamic_stop:.6f} -> {dynamic_stop:.6f}"
                        # )

                    # Determine stop loss price based on conditions
                    if use_dynamic_stop and dynamic_stop is not None:
                        # Use dynamic stop with ATR buffer
                        atr_stop_price = last_candle["close"] + trailing_atr
                        stop_loss_price = min(dynamic_stop * 1.005, atr_stop_price)

                        # Debug logging for final stop calculation
                        # logger.debug(
                        #     f"Using dynamic stop for {pair} (short): "
                        #     f"Dynamic={dynamic_stop:.6f}, ATR stop={atr_stop_price:.6f}, "
                        #     f"Final={stop_loss_price:.6f}"
                        # )
                    else:
                        return None

            # Convert to percentage
            if stop_loss_price > 0:
                final_stoploss = stoploss_from_absolute(
                    stop_loss_price,
                    current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )

                # Only log when there's an actual change in stop loss value
                # Use a small epsilon for floating point comparison
                epsilon = 0.0005  # Update stop loss only if the change is more than 0.05%
                current_stop_loss = trade.stop_loss if trade.stop_loss else 0

                # Check if the difference is significant (greater than epsilon)
                stop_loss_changed = (abs(stop_loss_price - current_stop_loss) / current_stop_loss) > epsilon

                if stop_loss_changed:
                    logger.debug(
                        f"{current_time.strftime('%Y-%m-%d %H:%M')} Stoploss update for {pair} "
                        f"({'short' if trade.is_short else 'long'}): "
                        f"price={stop_loss_price:.6f}, "
                        f"percent={final_stoploss:.4%}"
                    )
                else:
                    return None

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
        max_open_trades = self.config.get("max_open_trades", 1)
        if not isinstance(max_open_trades, int) or max_open_trades <= 0:
            logger.warning(
                f"Invalid max_open_trades value: {max_open_trades}. Defaulting to 1."
            )
            max_open_trades = 1

        if open_trades_count >= max_open_trades:
            return 0.0  # No slots available

        available_slots = max_open_trades - open_trades_count
        # This check should ideally not be needed if open_trades_count < max_open_trades
        # but as a safeguard:
        if available_slots <= 0:
            return 0.0

        collateral_per_slot = total_equity / available_slots
        logger.debug(
            f"collateral per slot: {collateral_per_slot} {total_equity} {available_slots}"
        )
        return collateral_per_slot

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:

        total_equity = self.get_total_equity()
        collateral_per_slot = self._get_collateral_per_trade_slot(total_equity)

        actual_stake_to_use = collateral_per_slot

        # Ensure stake is within min/max limits
        if min_stake is not None:
            actual_stake_to_use = max(actual_stake_to_use, min_stake)
        actual_stake_to_use = min(actual_stake_to_use, max_stake)

        logger.debug(
            f"Actual_stake_to_use ({pair}): {actual_stake_to_use} {collateral_per_slot}"
        )
        return actual_stake_to_use

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        entry_tag: str | None = None,
        **kwargs,
    ) -> float:
        """
        Calculate leverage based on maximum risk per trade.
        The goal is to size the position such that if the stop-loss is hit,
        the loss is no more than max_risk_per_trade of total equity.

        - Sets the maximum risk as a modifiable constant (max_risk_per_trade).
        - Risk in stake currency is (total_equity * max_risk_per_trade).
        - Desired position size (base currency) = risk_amount / (current_rate - stop_loss_price).
        - Calculated leverage = (desired_position_size * current_rate) / stake_for_this_trade.
        - If calculated leverage > max_leverage, do not enter (return 0.0).
        """
        # Get total equity in stake currency
        total_equity = self.get_total_equity()
        logger.debug(f"Leverage: Calculating total equity for {pair}: {total_equity}")

        if total_equity <= 1e-7:  # Effectively zero equity
            return 0.0  # Not enough equity to calculate leverage

        # Calculate risk amount in stake currency
        risk_amount_stake_curr = total_equity * self.max_risk_per_trade.value

        analyzed_df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if analyzed_df.empty:
            logger.warning(
                f"Leverage: Empty dataframe for pair {pair}, cannot determine stop-loss."
            )
            return 0.0  # Cannot determine stop loss, do not trade
        last_candle = analyzed_df.iloc[-1].squeeze()

        stop_loss_price = None
        price_diff_to_stop = 0.0

        # Determine stop loss based on entry type
        if entry_tag == "cradle":
            if side == "long":
                raw_stop_price = last_candle.get("stop_lower")
                if pd.isna(raw_stop_price):
                    logger.warning(
                        f"Leverage: stop_lower is np.nan for {pair} on {current_time}."
                    )
                    return 0.0  # Stop-loss level not found or np.nan
                stop_loss_price = raw_stop_price * 0.995
                if current_rate <= stop_loss_price:
                    return 0.0  # Invalid stop-loss for long
                price_diff_to_stop = current_rate - stop_loss_price
            elif side == "short":
                raw_stop_price = last_candle.get("stop_upper")
                if pd.isna(raw_stop_price):
                    logger.warning(
                        f"Leverage: stop_upper is np.nan for {pair} on {current_time}."
                    )
                    return 0.0  # Stop-loss level not found or np.nan
                stop_loss_price = raw_stop_price * 1.005
                if current_rate >= stop_loss_price:
                    return 0.0  # Invalid stop-loss for short
                price_diff_to_stop = stop_loss_price - current_rate
        elif entry_tag == "breakout":
            if side == "long":
                raw_stop_price = last_candle.get("close_minus_2atr")
                if pd.isna(raw_stop_price):
                    logger.warning(
                        f"Leverage: close_minus_2atr is np.nan for {pair} on {current_time}."
                    )
                    return 0.0  # Stop-loss level not found or np.nan
                stop_loss_price = raw_stop_price * 0.995
                if current_rate <= stop_loss_price:
                    return 0.0  # Invalid stop-loss for long
                price_diff_to_stop = current_rate - stop_loss_price
            elif side == "short":
                raw_stop_price = last_candle.get("close_plus_2atr")
                if pd.isna(raw_stop_price):
                    logger.warning(
                        f"Leverage: close_plus_2atr is np.nan for {pair} on {current_time}."
                    )
                    return 0.0  # Stop-loss level not found or np.nan
                stop_loss_price = raw_stop_price * 1.005
                if current_rate >= stop_loss_price:
                    return 0.0  # Invalid stop-loss for short
                price_diff_to_stop = stop_loss_price - current_rate
        else:  # Default to lrsi stop loss (or any other entry type)
            if side == "long":
                raw_stop_price = last_candle.get("long_stop")
                if pd.isna(raw_stop_price):
                    logger.warning(
                        f"Leverage: long_stop is np.nan for {pair} on {current_time}."
                    )
                    return 0.0  # Stop-loss level not found or np.nan
                stop_loss_price = raw_stop_price * 0.995
                if current_rate <= stop_loss_price:
                    return 0.0  # Invalid stop-loss for long
                price_diff_to_stop = current_rate - stop_loss_price
            elif side == "short":
                raw_stop_price = last_candle.get("short_stop")
                if pd.isna(raw_stop_price):
                    logger.warning(
                        f"Leverage: short_stop is np.nan for {pair} on {current_time}."
                    )
                    return 0.0  # Stop-loss level not found or np.nan
                stop_loss_price = raw_stop_price * 1.005
                if current_rate >= stop_loss_price:
                    return 0.0  # Invalid stop-loss for short
                price_diff_to_stop = stop_loss_price - current_rate

        if side not in ["long", "short"]:
            logger.error(f"Leverage: Invalid side '{side}' received.")
            return 0.0  # Should not happen

        if (
            price_diff_to_stop <= 1e-7
        ):  # Avoid division by zero or very small stop distance
            return 0.0  # Stop too close, do not enter

        # Desired position size in base currency
        desired_position_size_base = risk_amount_stake_curr / price_diff_to_stop
        # Desired position value in stake currency
        desired_position_value_stake_curr = desired_position_size_base * current_rate

        # Collateral Freqtrade would allocate for this trade slot by default.
        collateral_for_this_trade_slot = self._get_collateral_per_trade_slot(
            total_equity
        )

        if (
            collateral_for_this_trade_slot <= 1e-7
        ):  # Effectively zero collateral per slot
            return 0.0  # No collateral available per slot, do not trade

        required_leverage = (
            desired_position_value_stake_curr / collateral_for_this_trade_slot
        )

        if required_leverage <= 1e-7:  # Effectively zero or negative desired leverage
            return 0.0  # Do not trade

        if required_leverage < 1.0:
            return 0.0  # Do not trade, likely the opportunity is not worth it
        else:
            final_leverage = required_leverage

        # Ensure leverage is capped by max_leverage
        final_leverage = min(final_leverage, max_leverage)

        return float(round(final_leverage, 6))  # Round to a sensible precision

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        """
        Called right after an order fills.
        """
        logger.info(
            f"Order filled callback triggered for {pair}: "
            f"order_side={order.ft_order_side}, "
            f"order_type={order.order_type}"
        )

        # Exit if order is not an entry order
        if order.ft_order_side != trade.entry_side:
            # logger.info(f"Skipping non-entry order: {order.ft_order_side}")
            return None

        # Obtain pair dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Calculate take profit levels based on entry type
        self._set_trade_initial_values(
            trade, last_candle
        )

        return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
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

        take_profit_price = trade.get_custom_data(key="take_profit_price")
        take_profit_2_price = trade.get_custom_data(
            key="take_profit_2_price", default=None
        )
        take_profit_reduced = trade.get_custom_data(
            key="take_profit_reduced", default=False
        )
        take_profit_2_reduced = trade.get_custom_data(
            key="take_profit_2_reduced", default=False
        ) if self.use_take_profit_2.value and trade.enter_tag == "lrsi" else True

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
            trade.set_custom_data(key="take_profit_reduced", value=True)

            side_text = "short" if trade.is_short else "long"
            logger.info(
                f"Take profit reached for {trade.pair} ({side_text}) at {current_rate:.6f} "
                f"(target: {take_profit_price:.6f}). Reducing position by 50%."
            )

            # Calculate the correct stake amount to reduce position by exactly 50%
            # FreqTrade formula: amount_to_exit = abs(stake_amount) * trade.amount / trade.stake_amount
            # To exit 50% of position: 0.5 * trade.amount = abs(stake_amount) * trade.amount / trade.stake_amount
            # Solving: stake_amount = -0.5 * trade.stake_amount (negative for reduction)
            reduction_stake_amount = -0.5 * trade.stake_amount

            # Calculate expected amount to be exited for validation
            # expected_exit_amount = (
            #     abs(reduction_stake_amount) * trade.amount / trade.stake_amount
            # )
            # expected_exit_percentage = (expected_exit_amount / trade.amount) * 100

            # logger.debug(f"Position reduction calculation for {trade.pair}:")
            # logger.debug(
            #     f"  Current position: {trade.amount:.8f} {trade.base_currency}"
            # )
            # logger.debug(
            #     f"  Current stake: {trade.stake_amount:.6f} {trade.stake_currency}"
            # )
            # logger.debug(f"  Reduction stake amount: {reduction_stake_amount:.6f}")
            # logger.debug(
            #     f"  Expected exit amount: {expected_exit_amount:.8f} ({expected_exit_percentage:.1f}%)"
            # )

            return reduction_stake_amount

        take_profit_2_reached = False
        if take_profit_2_price is not None and not take_profit_2_reduced:
            if not trade.is_short:
                take_profit_2_reached = current_rate >= take_profit_2_price
            else:
                take_profit_2_reached = current_rate <= take_profit_2_price

        if take_profit_2_reached:
            # Mark that we've reduced the position at take profit 2
            trade.set_custom_data(key="take_profit_2_reduced", value=True)

            side_text = "short" if trade.is_short else "long"
            logger.info(
                f"Take profit 2 reached for {trade.pair} ({side_text}) at {current_rate:.6f} "
                f"(target: {take_profit_2_price:.6f}). Reducing position by 30% of original stake."
            )

            # To reduce by 30% of the original stake, we must reduce by 60% of the
            # remaining stake (since 50% was already sold).
            reduction_stake_amount = -0.6 * trade.stake_amount

            # Calculate expected amount to be exited for validation
            # expected_exit_amount = (
            #     abs(reduction_stake_amount) * trade.amount / trade.stake_amount
            # )

            # expected_exit_percentage = (expected_exit_amount / trade.amount) * 100

            # logger.debug(f"Position reduction calculation for {trade.pair}:")
            # logger.debug(
            #     f"  Current position: {trade.amount:.8f} {trade.base_currency}"
            # )
            # logger.debug(
            #     f"  Current stake: {trade.stake_amount:.6f} {trade.stake_currency}"
            # )
            # logger.debug(f"  Reduction stake amount: {reduction_stake_amount:.6f}")
            # logger.debug(
            #     f"  Expected exit amount: {expected_exit_amount:.8f} ({expected_exit_percentage:.1f}%)"
            # )

            return reduction_stake_amount

        # If we've already reduced at take profit, let the remaining position run
        return None

    def plot_annotations(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime,
        dataframe: DataFrame,
        **kwargs,
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
        peak_col = f"peak_{self.primary_timeframe}"
        trough_col = f"trough_{self.primary_timeframe}"
        upper_col = f"donchian_upper_{self.major_timeframe}"
        lower_col = f"donchian_lower_{self.major_timeframe}"

        if peak_col not in dataframe.columns or trough_col not in dataframe.columns:
            logger.warning(
                f"Peak/trough columns not found for {pair}. Available columns: {dataframe.columns.tolist()}"
            )
            return annotations

        # Filter dataframe to the requested date range
        df_filtered = dataframe[
            (dataframe["date"] >= start_date) & (dataframe["date"] <= end_date)
        ].copy()

        if df_filtered.empty:
            return annotations

        # Identify significant peak and trough changes
        df_filtered["peak_change"] = df_filtered[peak_col] != df_filtered[
            peak_col
        ].shift(1)
        df_filtered["trough_change"] = df_filtered[trough_col] != df_filtered[
            trough_col
        ].shift(1)
        df_filtered["significant_change"] = (
            df_filtered["peak_change"] | df_filtered["trough_change"]
        )

        # Always include start_date and end_date as transition points
        df_filtered.loc[df_filtered.index[0], "significant_change"] = (
            True  # First row (start_date)
        )
        df_filtered.loc[df_filtered.index[-1], "significant_change"] = (
            True  # Last row (end_date)
        )

        # Get transition points where peaks or troughs change
        transition_points = df_filtered[df_filtered["significant_change"]].copy()

        if len(transition_points) < 2:
            return annotations

        # Create ranges between transition points
        ranges = []
        for i in range(1, len(transition_points)):
            prev_point = transition_points.iloc[i - 1]
            current_point = transition_points.iloc[i]

            # Determine the relationship type for this range
            prev_peak = prev_point[peak_col]
            prev_trough = prev_point[trough_col]

            # Classify the range based on directional movement
            range_type = None

            # Classify based on overall market structure direction
            upswing_val = prev_point.get(f"ha_upswing_{self.major_timeframe}")
            downswing_val = prev_point.get(f"ha_downswing_{self.major_timeframe}")
            if upswing_val:
                range_type = "bullish"
                val = prev_point.get(upper_col)
                if val is not None:
                    prev_peak = val
            elif downswing_val:
                range_type = "bearish"
                val = prev_point.get(lower_col)
                if val is not None:
                    prev_trough = val
            else:
                # Fallback for edge cases
                range_type = "neutral"

            ranges.append(
                {
                    "start": prev_point["date"],
                    "end": current_point["date"],
                    "type": range_type,
                    "start_peak": prev_peak,
                    "start_trough": prev_trough
                }
            )

        # Create annotations from merged ranges
        for range_data in ranges:
            # Calculate y_start and y_end
            y_start = range_data["start_trough"]
            y_end = range_data["start_peak"]

            # Set colors based on market structure bias
            if range_data["type"] == "bullish":
                color = (
                    "rgba(144, 238, 144, 0.3)"  # Light green - for bullish structure
                )
            elif range_data["type"] == "bearish":
                color = (
                    "rgba(255, 182, 193, 0.3)"  # Light pink/red - for bearish structure
                )
            elif range_data["type"] == "neutral":
                color = "rgba(255, 255, 224, 0.3)"  # Light yellow - for neutral/consolidation
            else:
                # Fallback for any unexpected range type
                continue

            # Only create annotation if there's a meaningful price difference
            if (
                y_end > y_start and (y_end - y_start) / y_start > 0.001
            ):  # At least 0.1% difference
                annotations.append(
                    {
                        "type": "area",
                        # "label": label,
                        "start": range_data["start"],
                        "end": range_data["end"],
                        "y_start": y_start,
                        "y_end": y_end,
                        "color": color,
                    }
                )
            else:
                logger.debug(
                    f"Skipping annotation with insufficient price range: {y_start:.6f} - {y_end:.6f}"
                )

        logger.debug(
            f"Created {len(annotations)} market structure annotations for {pair}"
        )
        return annotations
