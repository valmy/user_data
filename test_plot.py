# Change directory
# Modify this cell to insure that the output shows the correct path.
# Define all paths relative to the project root shown in the cell output
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pprint

from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers import StrategyResolver
from freqtrade.exchange.binance import Binance


project_root = "/freqtrade"
i = 0
try:
    os.chdir(project_root)
    if not Path("LICENSE").is_file():
        i = 0
        while i < 4 and (not Path("LICENSE").is_file()):
            os.chdir(Path(Path.cwd(), "../"))
            i += 1
        project_root = Path.cwd()
except FileNotFoundError:
    print("Please define the project root relative to the current directory")

print(Path.cwd())

# Customize these according to your needs.

# Initialize empty configuration object
# config = Configuration.from_files([])
config = Configuration.from_files(["user_data/config.json"])

# Define some constants
config["timeframe"] = "5m"
# Name of the strategy class
config["strategy"] = "FractalStrategy"
# Location of the data
data_location = config["datadir"]

# Date range configuration
date_range_days = 1  # Duration of each date range (e.g., 2 = 2-day ranges like July 1-3, July 4-6)
overall_start = "2025-07-01"
overall_end = datetime.fromtimestamp(datetime.now().timestamp(), tz=timezone.utc).strftime("%Y-%m-%d")
date_ranges = []
current_date = datetime.strptime(overall_start, "%Y-%m-%d")
end_date_dt = datetime.strptime(overall_end, "%Y-%m-%d")

while current_date < end_date_dt:
    next_date = current_date + timedelta(days=date_range_days)
    if next_date > end_date_dt:
        next_date = end_date_dt
    date_ranges.append((
        current_date.strftime("%Y-%m-%d"),
        next_date.strftime("%Y-%m-%d")
    ))
    current_date = next_date + timedelta(days=1)  # Move to the day after next_date to avoid overlap
base_currency = "USDT" # Assuming USDT as the common quote and stake currency
stake_currency = "USDT"

# Initialize exchange first (this sets candle_type_def automatically)
exchange = Binance(config)

# Now load strategy using StrategyResolver (which properly handles the config)
loaded_strategy = StrategyResolver.load_strategy(config)
loaded_strategy.dp = DataProvider(config, exchange, None)
loaded_strategy.ft_bot_start()

# Get timeframes from the loaded strategy
primary_timeframe = loaded_strategy.primary_timeframe
major_timeframe = loaded_strategy.major_timeframe

# if backtest_dir points to a directory, it'll automatically load the last backtest file.
backtest_dir = config["user_data_dir"] / "backtest_results"
# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

# Load backtested trades as dataframe (once)
all_trades = load_backtest_data(backtest_dir)

pair = "ETH/USDT:USDT"
candles = load_pair_history(
    datadir=data_location, # This will be used inside the loop per pair
    timeframe=config["timeframe"], # This will be used inside the loop per pair
    pair=pair, # Placeholder, will be overridden in loop
    data_format="feather",
    candle_type=CandleType.FUTURES, # This will be used inside the loop per pair
) # This initial call might not be strictly necessary if immediately looped

print(candles.loc[
    (candles['date'] >= datetime(2025, 7, 15, 7, 30, tzinfo=timezone.utc)) &
    (candles['date'] <= datetime(2025, 7, 15, 8, 15, tzinfo=timezone.utc)),
    ['date', 'open', 'close', 'high', 'low', 'volume']
])


df = loaded_strategy.analyze_ticker(candles, {"pair": pair})


# Strategy and exchange are already loaded above

# print(candles)
print(df.loc[
    (df['date'] >= datetime(2025, 7, 15, 7, 30, tzinfo=timezone.utc)) &
    (df['date'] <= datetime(2025, 7, 15, 8, 15, tzinfo=timezone.utc)),
    ['date', 'high', 'low', 'trough_15m', 'peak_15m', 'donchian_upper_15m', 'donchian_lower_15m']
])
# print(all_trades)

start_date = datetime.strptime("2025-07-15", "%Y-%m-%d").replace(tzinfo=timezone.utc)
end_date = start_date + timedelta(days=1)

annotations = loaded_strategy.plot_annotations(
    pair=pair,
    start_date=start_date,
    end_date=end_date,
    dataframe=df,
)

pprint.pprint(annotations)

# print(all_trades.keys())
# print(all_trades.loc[:, ['close_date', 'pair', 'profit_abs', 'stop_loss_abs']])
