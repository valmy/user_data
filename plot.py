import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange, load_data
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper

logger = logging.getLogger(__name__)

project_root = "."
os.chdir(project_root)
if not Path("user_data/config-backtest.json").is_file():
    logger.error("Wrong path, config-backtest,json not found")
    sys.exit(1)

## Parse command line arguments
parser = argparse.ArgumentParser(description="Backtesting analysis script")
parser.add_argument(
    "--timerange",
    type=str,
    default="20250801-",
    help="Period for analysis (YYYYMMDD-YYYYMMDD format)",
)
args = parser.parse_args()

# Set timerange to use
timerange = TimeRange.parse_timerange(args.timerange)

# Initialize configuration object
config = Configuration.from_files(["user_data/config-backtest.json"])
exchange = ExchangeResolver.load_exchange(config)
strategy = StrategyResolver.load_strategy(config)
IStrategy.dp = DataProvider(config, exchange)

strategy.ft_bot_start()
strategy_safe_wrapper(strategy.bot_loop_start)(current_time=datetime.now(UTC))

markets = exchange.get_markets().keys()
if "pairs" in config:
    pairs = expand_pairlist(config["pairs"], markets)
else:
    pairs = expand_pairlist(config["exchange"]["pair_whitelist"], markets)

data = load_data(
    datadir=config.get("datadir"),
    pairs=pairs,
    timeframe=config["timeframe"],
    timerange=timerange,
    startup_candles=0,
    data_format=config["dataformat_ohlcv"],
    candle_type=config.get("candle_type_def"),
)

print(data['ETH/USDT:USDT'])
