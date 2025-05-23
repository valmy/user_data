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
config["strategy"] = "FractalStrategy"
# Location of the data
data_location = config["datadir"]

# start and end date
start_date = "2025-04-03"
end_date = "2025-04-03"
base_currency = "USDT" # Assuming USDT as the common quote and stake currency
stake_currency = "USDT"

# pairs_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'SUI', 'TRX', 'LINK']
pairs_symbols = ['SUI']

from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

# if backtest_dir points to a directory, it'll automatically load the last backtest file.
backtest_dir = config["user_data_dir"] / "backtest_results"
# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

# Load backtested trades as dataframe (once)
all_trades = load_backtest_data(backtest_dir)



# Load data using values set above
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType


# candles = load_pair_history(
#     datadir=data_location, # This will be used inside the loop per pair
#     timeframe=config["timeframe"], # This will be used inside the loop per pair
#     pair="ETH/USDT:USDT", # Placeholder, will be overridden in loop
#     data_format="feather",
#     candle_type=CandleType.FUTURES, # This will be used inside the loop per pair
# ) # This initial call might not be strictly necessary if immediately looped


# Load strategy and exchange once
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers import StrategyResolver
from freqtrade.exchange.binance import Binance

# print(config['exchange'])
strategy_name = config["strategy"] # Use the strategy name from config

loaded_strategy = StrategyResolver.load_strategy(config)
exchange = Binance(config) # Assuming Binance, adjust if necessary
loaded_strategy.dp = DataProvider(config, exchange, None)
loaded_strategy.ft_bot_start()


# All statistics are available per strategy, so if `--strategy-list` was used during backtest,
# this will be reflected here as well.
# Example usages:
print(stats["strategy"][strategy_name]["results_per_pair"])
# Get pairlist used for this backtest
print(stats["strategy"][strategy_name]["pairlist"])
# Get market change (average change of all pairs from start to end of the backtest period)
print(stats["strategy"][strategy_name]["market_change"])
# Maximum drawdown ()
print(stats["strategy"][strategy_name]["max_drawdown_abs"])
# Maximum drawdown start and end
print(stats["strategy"][strategy_name]["drawdown_start"])
print(stats["strategy"][strategy_name]["drawdown_end"])


# Get strategy comparison (only relevant if multiple strategies were compared)
print(stats["strategy_comparison"])

# Print all trades with date, pair, and profit details
print("\nAll Trades (from backtest results):")
if not all_trades.empty:
    print(all_trades[['open_date', 'pair', 'profit_ratio', 'profit_abs', 'exit_reason', 'trade_duration']])
else:
    print("No trades found in the backtest results.")

# Show value-counts per pair
if not all_trades.empty:
    print(all_trades.groupby("pair")["exit_reason"].value_counts())
else:
    print("No trades to group by.")

from freqtrade.plot.plotting import generate_candlestick_graph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

for pair_symbol in pairs_symbols:
    pair = f"{pair_symbol}/{base_currency}:{stake_currency}"
    print(f"\nProcessing pair: {pair}")
    print(f"Data location: {data_location}")

    # Load data using values set above
    candles = load_pair_history(
        datadir=data_location,
        timeframe=config["timeframe"],
        pair=pair,
        data_format="feather",  # Make sure to update this to your data
        candle_type=CandleType.FUTURES,
    )

    # Confirm success
    if candles.empty:
        print(f"No data found for {pair} from {data_location}. Skipping.")
        continue
    print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")
    # print(candles)

    # Generate buy/sell signals using strategy
    df = loaded_strategy.analyze_ticker(candles, {"pair": pair})
    # print(df)

    print(f"Generated {df['enter_long'].sum()} long entry signals for {pair}")
    print(f"Generated {df['enter_short'].sum()} short entry signals for {pair}")
    data = df.set_index("date", drop=False)
    # print(data)

    # Limit graph period to keep plotly quick and reactive

    # a day after end date
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    end_date_plus_dt = end_date_dt + timedelta(days=1)
    end_date_plus = end_date_plus_dt.strftime("%Y-%m-%d")

    # Filter trades to the current pair and date range
    if not all_trades.empty:
        trades_red = all_trades.loc[all_trades["pair"] == pair]
        trades_red = trades_red[(trades_red["open_date"] >= start_date) & (trades_red["open_date"] < end_date_plus)]
    else:
        trades_red = pd.DataFrame() # Empty dataframe if no trades

    data_red = data[start_date:end_date]

    # Print filtered trades (trades_red) for the plot
    print(f"\nFiltered Trades for {pair} (trades_red for plot):")
    if not trades_red.empty:
        print(trades_red[['open_date', 'pair', 'is_short', 'open_rate', 'profit_ratio',
                          'profit_abs', 'exit_reason', 'trade_duration']])
        # Print corresponding data points from data_red
        print("\nCorresponding Data Points for Trades:")
        for index, trade in trades_red.iterrows():
            trade_date = trade['open_date']
            # Find the closest data point in data_red
            closest_data_point = data_red.iloc[
                data_red.index.get_loc(trade_date)
            ]
            print(f"Trade Date: {trade_date}, "
                  f"Open: {closest_data_point['open']:.8f}, "
                  f"Close: {closest_data_point['close']:.8f}, "
                  f"High: {closest_data_point['high']:.8f}, "
                  f"Low: {closest_data_point['low']:.8f}, "
                  f"Chop 15m: {closest_data_point.get('chop_15m', 'N/A'):.2f}, "
                  f"Chop 1h: {closest_data_point.get('chop_1h', 'N/A'):.2f}, "
                  f"Enter Long: {closest_data_point.get('enter_long', 'N/A')}, "
                  f"Enter Short: {closest_data_point.get('enter_short', 'N/A')}")

    else:
        print(f"No trades found for {pair} in the specified date range for plotting.")


    # Create the candlestick chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2]) # Adjusted for 3 rows

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

    # Add Major Peak line
    if 'peak_1h' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['peak_1h'],
            name='Peak',
            line=dict(color=px.colors.qualitative.Pastel[0], width=1)
        ), row=1, col=1)

    # Add Major Trough line
    if 'trough_1h' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['trough_1h'],
            name='Trough',
            line=dict(color=px.colors.qualitative.Pastel[1], width=1)
        ), row=1, col=1)

    # Add Primary Peak line (from 15m timeframe)
    if 'peak_15m' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['peak_15m'],
            name='Primary Peak (15m)',
            line=dict(color=px.colors.qualitative.Pastel[2], width=1)
        ), row=1, col=1)

    # Add Primary Trough line (from 15m timeframe)
    if 'trough_15m' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['trough_15m'],
            name='Primary Trough (15m)',
            line=dict(color=px.colors.qualitative.Pastel[3], width=1)
        ), row=1, col=1)

    # Add markers for higher_high_1h at peak_1h
    # Add markers for higher_high_1h at peak_1h, only at the start of each hour
    if 'higher_high_1h' in data_red.columns and 'peak_1h' in data_red.columns and not data_red.empty:
        # Filter data to the first entry of each hour where higher_high_1h is True
        hh_data = data_red[(data_red['higher_high_1h'] == True) & (data_red.index.minute == 0)]
        if not hh_data.empty:
            fig.add_trace(go.Scatter(
                x=hh_data.date,
                y=hh_data['peak_1h'],
                mode='markers',
                name='Higher High (1h)',
                marker=dict(symbol='triangle-up', color=px.colors.qualitative.Pastel[4]),
                hoverinfo='skip'
            ), row=1, col=1)

    # Add markers for lower_low_1h at trough_1h
    # Add markers for lower_low_1h at trough_1h, only at the start of each hour
    if 'lower_low_1h' in data_red.columns and 'trough_1h' in data_red.columns and not data_red.empty:
        # Filter data to the first entry of each hour where lower_low_1h is True
        ll_data = data_red[(data_red['lower_low_1h'] == True) & (data_red.index.minute == 0)]
        if not ll_data.empty:
            fig.add_trace(go.Scatter(
                x=ll_data.date,
                y=ll_data['trough_1h'],
                mode='markers',
                name='Lower Low (1h)',
                marker=dict(symbol='triangle-down', color=px.colors.qualitative.Pastel[5]),
                hoverinfo='skip'
            ), row=1, col=1)

    # Add Laguerre RSI line
    if 'laguerre' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['laguerre'],
            name='Laguerre RSI',
            line=dict(color='yellow', width=1)
        ), row=2, col=1)

    # Add horizontal lines at 0.2 and 0.8 for Laguerre RSI
    fig.add_hline(y=0.8, line_dash="dash", row=2, col=1,
                annotation_text="Overbought (0.8)",
                annotation_position="bottom right",
                line_color="rgba(200, 200, 200, 0.5)")
    fig.add_hline(y=0.2, line_dash="dash", row=2, col=1,
                annotation_text="Oversold (0.2)",
                annotation_position="bottom right",
                line_color="rgba(200, 200, 200, 0.5)")

    # Original volume bar code (commented out or removed)
    """fig.add_trace(go.Bar(
        x=data_red.date,
        y=data_red.volume,
        name="Volume",
        marker_color='#5C6BC0',  # Blue for volume bars
    ), row=2, col=1)"""

    # Add Choppiness Index for 15m
    if 'chop_15m' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['chop_15m'],
            name='Chop 15m',
            line=dict(color='orange', width=1)
        ), row=3, col=1)

    # Add Choppiness Index for 1h
    if 'chop_1h' in data_red.columns:
        fig.add_trace(go.Scatter(
            x=data_red.date,
            y=data_red['chop_1h'],
            name='Chop 1h',
            line=dict(color='purple', width=1)
        ), row=3, col=1)

    # Update layout for a dark theme
    fig.update_layout(
        template='plotly_dark',
        title=f"Price Chart for {pair}",
        yaxis_title="Price (USD)",
        yaxis2_title="Laguerre RSI",
        yaxis3_title="Choppiness Index",
        showlegend=False,
        # Explicitly control the x-axis range slider
        # Set to True to ensure it's visible (default for candlestick)
        # Set to False to hide it
        xaxis_rangeslider_visible=False
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
            # Separate long and short entries
            long_trades = trades[trades["is_short"] == False]
            short_trades = trades[trades["is_short"] == True]

            long_trade_entries = go.Scatter(
                x=long_trades["open_date"],
                y=long_trades["open_rate"],
                mode="markers",
                name="Long Entry",
                text=long_trades["desc"],
                marker=dict(symbol="triangle-up-open", size=11, line=dict(width=2), color="cyan"),
            )

            short_trade_entries = go.Scatter(
                x=short_trades["open_date"],
                y=short_trades["open_rate"],
                mode="markers",
                name="Short Entry",
                text=short_trades["desc"],
                marker=dict(symbol="triangle-down-open", size=11, line=dict(width=2), color="cyan"),
            )

            # Exits remain the same, or you can also differentiate them if needed
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
            fig.add_trace(long_trade_entries, 1, 1)
            fig.add_trace(short_trade_entries, 1, 1)
            fig.add_trace(trade_exits, 1, 1)
            fig.add_trace(trade_exits_loss, 1, 1)
        else:
            print("No trades found.")
        return fig

    plot_trades(fig, trades_red)

    # Customize grid
    fig.update_xaxes(gridcolor='#1f1f1f', zerolinecolor='#1f1f1f')
    fig.update_yaxes(gridcolor='#1f1f1f', zerolinecolor='#1f1f1f')

    # Save the interactive HTML chart
    html_filename = f"{project_root}/user_data/chart_5m_{pair_symbol}.html"
    png_filename = f"{project_root}/user_data/chart_5m_{pair_symbol}.png"

    fig.write_html(html_filename)
    print(f"Interactive chart for {pair} has been saved to {html_filename}")

    # Save a static PNG version
    fig.write_image(png_filename)
    print(f"Static chart for {pair} has been saved to {png_filename}")
