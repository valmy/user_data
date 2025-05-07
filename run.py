import os
from pathlib import Path


# Change directory
# Modify this cell to insure that the output shows the correct path.
# Define all paths relative to the project root shown in the cell output
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

from freqtrade.configuration import Configuration

# Customize these according to your needs.

# Initialize empty configuration object
# config = Configuration.from_files([])
# Optionally (recommended), use existing configuration file
config = Configuration.from_files(["user_data/config.json"])

# Define some constants
config["timeframe"] = "5m"
# Name of the strategy class
config["strategy"] = "DowTheoryStrategy"
# Location of the data
data_location = config["datadir"]
# Pair to analyze - Only use one pair here
pair = "BTC/USDT:USDT"

print(data_location)

# Load data using values set above
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType


candles = load_pair_history(
    datadir=data_location,
    timeframe=config["timeframe"],
    pair=pair,
    data_format="feather",  # Make sure to update this to your data
    candle_type=CandleType.FUTURES,
)

# Confirm success
print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")
print(candles)
# candles.tail()

# Load strategy using values set above
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers import StrategyResolver
from freqtrade.exchange.binance import Binance

print(config['exchange'])
strategy = StrategyResolver.load_strategy(config)
exchange = Binance(config)
strategy.dp = DataProvider(config, exchange, None)
strategy.ft_bot_start()

# Generate buy/sell signals using strategy
df = strategy.analyze_ticker(candles, {"pair": pair})
print(df)

print(f"Generated {df['enter_long'].sum()} long entry signals")
print(f"Generated {df['enter_short'].sum()} short entry signals")
data = df.set_index("date", drop=False)
print(data)


from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

# if backtest_dir points to a directory, it'll automatically load the last backtest file.
backtest_dir = config["user_data_dir"] / "backtest_results"
# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

strategy = "DowTheoryStrategy"
# All statistics are available per strategy, so if `--strategy-list` was used during backtest,
# this will be reflected here as well.
# Example usages:
print(stats["strategy"][strategy]["results_per_pair"])
# Get pairlist used for this backtest
print(stats["strategy"][strategy]["pairlist"])
# Get market change (average change of all pairs from start to end of the backtest period)
print(stats["strategy"][strategy]["market_change"])
# Maximum drawdown ()
print(stats["strategy"][strategy]["max_drawdown_abs"])
# Maximum drawdown start and end
print(stats["strategy"][strategy]["drawdown_start"])
print(stats["strategy"][strategy]["drawdown_end"])


# Get strategy comparison (only relevant if multiple strategies were compared)
print(stats["strategy_comparison"])

# Load backtested trades as dataframe
trades = load_backtest_data(backtest_dir)

# Show value-counts per pair
trades.groupby("pair")["exit_reason"].value_counts()

from freqtrade.plot.plotting import generate_candlestick_graph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd

# Limit graph period to keep plotly quick and reactive

# Filter trades to one pair and only May 1st
trades_red = trades.loc[trades["pair"] == pair]
trades_red = trades_red[(trades_red["open_date"] >= "2025-05-01") & (trades_red["open_date"] < "2025-05-02")]

data_red = data["2025-05-01":"2025-05-01"]

# Create the candlestick chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3])

candles = go.Candlestick(
    x=data_red.date,
    open=data_red.open,
    high=data_red.high,
    low=data_red.low,
    close=data_red.close,
    name="Price",
    increasing_line_color='#26A69A',    # Green for up candles
    decreasing_line_color='#EF5350'     # Red for down candles
)
fig.add_trace(candles, 1, 1)

# Donchian peaks
peaks = data_red[data_red['donchian_peak']]
fig.add_trace(go.Scatter(
    x=peaks['date'],
    y=peaks['high'],
    mode='markers',
    marker=dict(symbol='triangle-up', color='lime', size=12),
    name='Donchian Peak'
), row=1, col=1)

# Donchian troughs
troughs = data_red[data_red['donchian_trough']]
fig.add_trace(go.Scatter(
    x=troughs['date'],
    y=troughs['low'],
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=12),
    name='Donchian Trough'
), row=1, col=1)

# Add volume bars
fig.add_trace(go.Bar(
    x=data_red.date,
    y=data_red.volume,
    name="Volume",
    marker_color='#5C6BC0'  # Blue for volume bars
), row=2, col=1)

# Update layout for a dark theme
fig.update_layout(
    template='plotly_dark',
    title=f"Price Chart",
    yaxis_title="Price (USD)",
    yaxis2_title="Volume (USD)",
    showlegend=False
)

def plot_trades(fig, trades: pd.DataFrame) -> make_subplots:
    """
    Add trades to "fig"
    """
    # Trades can be empty
    if trades is not None and len(trades) > 0:
        # Create description for exit summarizing the trade
        trades["desc"] = trades.apply(
            lambda row: f"{row['profit_ratio']:.2%}, "
            + (f"{row['enter_tag']}, " if row["enter_tag"] is not None else "")
            + f"{row['exit_reason']}, "
            + f"{row['trade_duration']} min",
            axis=1,
        )
        trade_entries = go.Scatter(
            x=trades["open_date"],
            y=trades["open_rate"],
            mode="markers",
            name="Trade entry",
            text=trades["desc"],
            marker=dict(symbol="circle-open", size=11, line=dict(width=2), color="cyan"),
        )

        trade_exits = go.Scatter(
            x=trades.loc[trades["profit_ratio"] > 0, "close_date"],
            y=trades.loc[trades["profit_ratio"] > 0, "close_rate"],
            text=trades.loc[trades["profit_ratio"] > 0, "desc"],
            mode="markers",
            name="Exit - Profit",
            marker=dict(symbol="square-open", size=11, line=dict(width=2), color="green"),
        )
        trade_exits_loss = go.Scatter(
            x=trades.loc[trades["profit_ratio"] <= 0, "close_date"],
            y=trades.loc[trades["profit_ratio"] <= 0, "close_rate"],
            text=trades.loc[trades["profit_ratio"] <= 0, "desc"],
            mode="markers",
            name="Exit - Loss",
            marker=dict(symbol="square-open", size=11, line=dict(width=2), color="red"),
        )
        fig.add_trace(trade_entries, 1, 1)
        fig.add_trace(trade_exits, 1, 1)
        fig.add_trace(trade_exits_loss, 1, 1)
    else:
        print("No trades found.")
    return fig

plot_trades(fig, trades_red)

# Customize grid
fig.update_xaxes(gridcolor='#1f1f1f', zerolinecolor='#1f1f1f')
fig.update_yaxes(gridcolor='#1f1f1f', zerolinecolor='#1f1f1f')

# Get current date for filenames
current_date = datetime.now().strftime("%Y%m%d")

# Save the interactive HTML chart
html_filename = f"{project_root}/user_data/chart_5m_{current_date}.html"
png_filename = f"{project_root}/user_data/chart_5m_{current_date}.png"

fig.write_html(html_filename)
print(f"Interactive chart has been saved to {html_filename}")

# Save a static PNG version
fig.write_image(png_filename)
print(f"Static chart has been saved to {png_filename}")
