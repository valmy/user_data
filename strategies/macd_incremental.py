import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import find_peaks
import time

class MacdStatefulBot:
    """
    A stateful trading bot that detects MACD/Price convergence and divergence
    efficiently for live trading.

    It initializes its state once and then updates and analyzes incrementally
    with each new candle, achieving O(1) complexity per tick.
    """
    def __init__(self, historical_df: pd.DataFrame, lookback_window: int = 3):
        """
        Initializes the bot with historical data to establish the initial state.

        Args:
            historical_df (pd.DataFrame): A DataFrame with 'high', 'low', 'close' columns.
            lookback_window (int): How many candles to look back to confirm a peak/trough.
                                   A value of 1 means we check `df.iloc[-2]`.
        """
        print("--- Bot Initializing ---")
        if lookback_window < 1:
            raise ValueError("lookback_window must be at least 1.")

        self.lookback_pos = -1 - lookback_window # Position to check for peak/trough, e.g., -2
        self.df = historical_df.copy()

        # --- State Variables ---
        self.last_price_peak = {'index': -1, 'value': -np.inf}
        self.last_price_trough = {'index': -1, 'value': np.inf}
        self.last_macd_peak = {'index': -1, 'value': -np.inf}
        self.last_macd_trough = {'index': -1, 'value': np.inf}

        self._calculate_macd()
        self._initialize_state()
        print("--- Bot Initialized and Ready ---")

    def _calculate_macd(self):
        """Calculates or updates the MACD for the entire internal DataFrame."""
        # pandas-ta is efficient; recalculating is fine for this example.
        # For extreme performance, you could calculate only the last MACD value.
        self.df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def _initialize_state(self):
        """
        Analyzes the initial historical data ONCE to find the most recent
        peaks and troughs to set the initial state.
        """
        print("Finding initial peaks and troughs from historical data...")

        # Find all historical peaks and troughs
        price_peaks, _ = find_peaks(self.df['high'], distance=5, prominence=1)
        price_troughs, _ = find_peaks(-self.df['low'], distance=5, prominence=1)
        macd_peaks, _ = find_peaks(self.df['MACD_12_26_9'], distance=5, prominence=0.5)
        macd_troughs, _ = find_peaks(-self.df['MACD_12_26_9'], distance=5, prominence=0.5)

        # Set state to the most recent ones
        if len(price_peaks) > 0:
            idx = price_peaks[-1]
            self.last_price_peak = {'index': idx, 'value': self.df.at[idx, 'high']}
        if len(price_troughs) > 0:
            idx = price_troughs[-1]
            self.last_price_trough = {'index': idx, 'value': self.df.at[idx, 'low']}
        if len(macd_peaks) > 0:
            idx = macd_peaks[-1]
            self.last_macd_peak = {'index': idx, 'value': self.df.at[idx, 'MACD_12_26_9']}
        if len(macd_troughs) > 0:
            idx = macd_troughs[-1]
            self.last_macd_trough = {'index': idx, 'value': self.df.at[idx, 'MACD_12_26_9']}

        print(f"Initial Last Price Peak: Index={self.last_price_peak['index']}, Value={self.last_price_peak['value']:.2f}")
        print(f"Initial Last Price Trough: Index={self.last_price_trough['index']}, Value={self.last_price_trough['value']:.2f}")

    def on_new_candle(self, new_candle: pd.Series):
        """
        The main entry point for the live trading loop.

        Args:
            new_candle (pd.Series): A series with 'high', 'low', 'close' data.
        """
        # Append new data
        self.df = pd.concat([self.df, new_candle.to_frame().T], ignore_index=True)

        # Update MACD
        self._calculate_macd()

        # --- Core Stateful Logic ---
        # Check if the *previous* candle has now been confirmed as a peak or trough
        self._check_for_new_peak()
        self._check_for_new_trough()

    def _check_for_new_peak(self):
        """Checks if a new peak was just confirmed."""
        # A peak is confirmed at `lookback_pos` if it's higher than its neighbors
        pos = self.lookback_pos
        if (self.df['high'].iloc[pos] > self.df['high'].iloc[pos-1] and
            self.df['high'].iloc[pos] > self.df['high'].iloc[pos+1]):

            # New price peak found!
            new_price_peak_idx = self.df.index[pos]
            new_price_peak_val = self.df['high'].iloc[pos]

            # Avoid re-detecting the same peak
            if new_price_peak_idx == self.last_price_peak['index']:
                return

            print(f"\n📈 New Price Peak Confirmed at index {new_price_peak_idx}: {new_price_peak_val:.2f}")

            # Find the corresponding MACD value at this new peak
            new_macd_peak_val = self.df['MACD_12_26_9'].iloc[pos]

            # --- Analysis ---
            is_price_hh = new_price_peak_val > self.last_price_peak['value']
            is_macd_hh = new_macd_peak_val > self.last_macd_peak['value']

            if is_price_hh and is_macd_hh:
                print(">>> ✅ CONFIRMED: Bullish Convergence (Price HH, MACD HH)")
            elif is_price_hh and not is_macd_hh:
                print(">>> ⚠️ SIGNAL: Bearish Divergence (Price HH, MACD LH)")

            # --- Update State ---
            self.last_price_peak = {'index': new_price_peak_idx, 'value': new_price_peak_val}
            self.last_macd_peak = {'index': new_price_peak_idx, 'value': new_macd_peak_val}

    def _check_for_new_trough(self):
        """Checks if a new trough was just confirmed."""
        pos = self.lookback_pos
        if (self.df['low'].iloc[pos] < self.df['low'].iloc[pos-1] and
            self.df['low'].iloc[pos] < self.df['low'].iloc[pos+1]):

            new_price_trough_idx = self.df.index[pos]
            new_price_trough_val = self.df['low'].iloc[pos]

            if new_price_trough_idx == self.last_price_trough['index']:
                return

            print(f"\n📉 New Price Trough Confirmed at index {new_price_trough_idx}: {new_price_trough_val:.2f}")

            new_macd_trough_val = self.df['MACD_12_26_9'].iloc[pos]

            is_price_ll = new_price_trough_val < self.last_price_trough['value']
            is_macd_ll = new_macd_trough_val < self.last_macd_trough['value']

            if is_price_ll and is_macd_ll:
                print(">>> ✅ CONFIRMED: Bearish Convergence (Price LL, MACD LL)")
            elif is_price_ll and not is_macd_ll:
                print(">>> ⚠️ SIGNAL: Bullish Divergence (Price LL, MACD HL)")

            self.last_price_trough = {'index': new_price_trough_idx, 'value': new_price_trough_val}
            self.last_macd_trough = {'index': new_price_trough_idx, 'value': new_macd_trough_val}


# --- Simulation of a Live Environment ---
if __name__ == "__main__":
    # 1. Create sample data
    data = {
        'open':  [100, 102, 105, 103, 106, 108, 115, 112, 110, 113, 116, 122, 118, 117, 120, 125, 132, 128, 126, 129, 133, 130, 125, 122, 119],
        'high':  [103, 106, 107, 105, 109, 110, 118, 114, 112, 117, 120, 125, 120, 119, 124, 129, 135, 130, 128, 132, 138, 132, 127, 124, 120], # Peak at 16 (135) and 20 (138)
        'low':   [99,  101, 102, 101, 104, 107, 111, 111, 109, 111, 115, 117, 116, 115, 118, 123, 127, 127, 125, 128, 131, 129, 124, 121, 118], # Trough at 23 (121)
        'close': [102, 105, 106, 104, 108, 109, 116, 113, 111, 116, 119, 123, 117, 118, 122, 128, 132, 129, 127, 131, 136, 130, 126, 122, 119],
    }
    full_df = pd.DataFrame(data)

    # 2. Split data into initial history and a "live feed"
    initial_history = full_df.iloc[:20]
    live_feed = full_df.iloc[20:]

    # 3. Initialize the bot
    # We use lookback_window=1, so it checks the candle at index -2
    bot = MacdStatefulBot(historical_df=initial_history, lookback_window=1)

    # 4. Simulate the live feed
    print("\n--- Starting Live Feed Simulation ---\n")
    for index, new_candle in live_feed.iterrows():
        print(f"--- New Candle Received (Index {index}) ---")
        bot.on_new_candle(new_candle)
        time.sleep(0.5) # Pause to make the simulation readable

