import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.data.dataprovider import DataProvider
from freqtrade.resolvers import StrategyResolver
from freqtrade.exchange.binance import Binance
from strategies.FractalStrategy import FractalStrategy


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

# Date range configuration
from datetime import datetime, timedelta
overall_start = "2025-04-01"
overall_end = "2025-06-13"
date_ranges = []
current_date = datetime.strptime(overall_start, "%Y-%m-%d")
end_date_dt = datetime.strptime(overall_end, "%Y-%m-%d")

while current_date < end_date_dt:
    next_date = current_date + timedelta(days=3)
    if next_date > end_date_dt:
        next_date = end_date_dt
    date_ranges.append((
        current_date.strftime("%Y-%m-%d"),
        next_date.strftime("%Y-%m-%d")
    ))
    current_date = next_date
base_currency = "USDT" # Assuming USDT as the common quote and stake currency
stake_currency = "USDT"

pairs_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'SUI', 'TRX', 'LINK']
# pairs_symbols = ['ETH', 'XRP', 'LINK']

# Initialize strategy and get timeframes
strategy = FractalStrategy(config=config)
primary_timeframe = strategy.primary_timeframe
major_timeframe = strategy.major_timeframe

# if backtest_dir points to a directory, it'll automatically load the last backtest file.
backtest_dir = config["user_data_dir"] / "backtest_results"
# You can get the full backtest statistics by using the following command.
# This contains all information used to generate the backtest result.
stats = load_backtest_stats(backtest_dir)

# Load backtested trades as dataframe (once)
all_trades = load_backtest_data(backtest_dir)



# candles = load_pair_history(
#     datadir=data_location, # This will be used inside the loop per pair
#     timeframe=config["timeframe"], # This will be used inside the loop per pair
#     pair="ETH/USDT:USDT", # Placeholder, will be overridden in loop
#     data_format="feather",
#     candle_type=CandleType.FUTURES, # This will be used inside the loop per pair
# ) # This initial call might not be strictly necessary if immediately looped


# Load strategy and exchange once
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

for start_date, end_date in date_ranges:
    for pair_symbol in pairs_symbols:
        pair = f"{pair_symbol}/{base_currency}:{stake_currency}"
        try:
            # Load data
            candles = load_pair_history(
                datadir=data_location,
                timeframe=config["timeframe"],
                pair=pair,
                data_format="feather",
                candle_type=CandleType.FUTURES,
            )

            if candles.empty:
                print(f"No data found for {pair} from {data_location}. Skipping.")
                continue

            print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")

            # Generate signals
            df = loaded_strategy.analyze_ticker(candles, {"pair": pair})
            data = df.set_index("date", drop=False)

            # Filter trades for current date range
            end_date_plus = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            trades_red = all_trades[
                (all_trades["pair"] == pair) &
                (all_trades["open_date"] >= start_date) &
                (all_trades["open_date"] < end_date_plus)
            ] if not all_trades.empty else pd.DataFrame()

            data_red = data[start_date:end_date]

            # Create chart
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.6, 0.2, 0.2])

            # Add chart elements (candles, indicators, trades etc.)
            # ... (preserve existing chart configuration code here) ...

            # Generate HTML filename
            html_filename = f"{project_root}/user_data/chart_5m_{pair_symbol}_{start_date}_{end_date}.html"
            fig.write_html(html_filename)

            # Add navigation controls
            with open(html_filename, 'r+') as f:
                content = f.read()
                body_index = content.find('<body>') + 6
                nav_html = f'''
                <div style="padding: 10px; background: #1f1f1f; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        {f'<a href="chart_5m_{pair_symbol}_{(datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")}_{start_date}.html" style="color: white; text-decoration: none; padding: 5px 10px; border: 1px solid #666; border-radius: 4px;">← Previous</a>'
                        if (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)) >= datetime.strptime(overall_start, "%Y-%m-%d")
                        else '<span style="color: #666; padding: 5px 10px;">← Start</span>'}
                        <span style="color: #888; margin: 0 15px;">{start_date} to {end_date}</span>
                        {f'<a href="chart_5m_{pair_symbol}_{end_date}_{(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")}.html" style="color: white; text-decoration: none; padding: 5px 10px; border: 1px solid #666; border-radius: 4px;">Next →</a>'
                        if (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)) <= datetime.strptime(overall_end, "%Y-%m-%d")
                        else '<span style="color: #666; padding: 5px 10px;">End →</span>'}
                    </div>
                    <select onchange="window.location.href=this.value.replace('ASSET',this.options[this.selectedIndex].text)"
                            style="padding: 5px; background: #333; color: white; border: 1px solid #666; border-radius: 4px;">
                        <option value="">Select Asset</option>
                        {''.join(f'<option value="chart_5m_ASSET_{start_date}_{end_date}.html">{sym}</option>' for sym in pairs_symbols)}
                    </select>
                </div>
                '''
                modified_content = content[:body_index] + nav_html + content[body_index:]
                f.seek(0)
                f.write(modified_content)
                f.truncate()

            print(f"Successfully created chart: {html_filename}")

        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            continue

            candles = load_pair_history(
                datadir=data_location,
                timeframe=config["timeframe"],
                pair=pair,
                data_format="feather",
                candle_type=CandleType.FUTURES,
            )

            # Confirm success
            if candles.empty:
                print(f"No data found for {pair} from {data_location}. Skipping.")
                continue
            print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")

            # Generate buy/sell signals using strategy
            df = loaded_strategy.analyze_ticker(candles, {"pair": pair})
            print(f"Generated {df['enter_long'].sum()} long entry signals for {pair}")
            print(f"Generated {df['enter_short'].sum()} short entry signals for {pair}")
            data = df.set_index("date", drop=False)

            # Limit graph period to keep plotly quick and reactive
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
            end_date_plus_dt = end_date_dt + timedelta(days=1)
            end_date_plus = end_date_plus_dt.strftime("%Y-%m-%d")

            # Filter trades to the current pair and date range
            if not all_trades.empty:
                trades_red = all_trades.loc[all_trades["pair"] == pair]
                trades_red = trades_red[(trades_red["open_date"] >= start_date) & (trades_red["open_date"] < end_date_plus)]
            else:
                trades_red = pd.DataFrame()

            data_red = data[start_date:end_date]

            # Print filtered trades (trades_red) for the plot
            print(f"\nFiltered Trades for {pair} (trades_red for plot):")
            if not trades_red.empty:
                print(trades_red[['open_date', 'pair', 'is_short', 'open_rate', 'close_rate', 'profit_abs',
                                'profit_ratio', 'exit_reason', 'stake_amount', 'amount', 'leverage']])
                print("\nCorresponding Data Points for Trades:")
                for index, trade in trades_red.iterrows():
                    trade_date = trade['open_date']
                    closest_data_point = data_red.iloc[data_red.index.get_loc(trade_date)]
                    print(f"Trade Date: {trade_date}, "
                        f"Open: {closest_data_point['open']:.8f}, "
                        f"Close: {closest_data_point['close']:.8f}, "
                        f"High: {closest_data_point['high']:.8f}, "
                        f"Low: {closest_data_point['low']:.8f}")
                    close_date = trade['close_date']
                    if close_date in data_red.index:
                        closest_data_point = data_red.iloc[data_red.index.get_loc(close_date)]
                        print(f"Close Date: {trade['close_date']}"
                            f"O: {closest_data_point['open']:.4f}, "
                            f"C: {closest_data_point['close']:.4f}, "
                            f"H: {closest_data_point['high']:.4f}, "
                            f"L: {closest_data_point['low']:.4f}")
                    else:
                        print(f"Close Date: {trade['close_date']} not in data_red")
            else:
                print(f"No trades found for {pair} in the specified date range for plotting.")

            # Create and save chart
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.6, 0.2, 0.2])

            # ... (rest of chart creation code remains the same) ...

            # Save the interactive HTML chart
            html_filename = f"{project_root}/user_data/chart_5m_{pair_symbol}_{start_date}_{end_date}.html"
            fig.write_html(html_filename)

            # Add navigation and asset selector
            with open(html_filename, 'r+') as f:
                content = f.read()
                body_index = content.find('<body>') + 6
                # ... (navigation HTML insertion code remains the same) ...
                f.truncate()

            print(f"Interactive chart for {pair} has been saved to {html_filename}")

        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            continue
        candles = load_pair_history(
            datadir=data_location,
            timeframe=config["timeframe"],
            pair=pair,
            data_format="feather",
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
            print(trades_red[['open_date', 'pair', 'is_short', 'open_rate', 'close_rate', 'profit_abs',
                            'profit_ratio', 'exit_reason', 'stake_amount', 'amount', 'leverage']])
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
                  f"Trough 15m: {closest_data_point.get('trough_15m', 'N/A'):.2f}, "
                  f"Chop 15m: {closest_data_point.get('chop_15m', 'N/A'):.2f}, "
                  f"Chop 1h: {closest_data_point.get('chop_1h', 'N/A'):.2f}, "
                  f"Enter Long: {closest_data_point.get('enter_long', 'N/A')}, "
                  f"Enter Short: {closest_data_point.get('enter_short', 'N/A')}")
            close_date = trade['close_date']
            # if close_date is not in data_red, print the trade
            if close_date not in data_red.index:
                print(f"Close Date: {trade['close_date']} not in data_red")
                continue
            # Find the closest data point in data_red
            closest_data_point = data_red.iloc[
                data_red.index.get_loc(close_date)
            ]
            print(f"Close Date: {trade['close_date']}"
                  f"O: {closest_data_point['open']:.4f}, "
                  f"C: {closest_data_point['close']:.4f}, "
                  f"H: {closest_data_point['high']:.4f}, "
                  f"L: {closest_data_point['low']:.4f}, "
                  f"peak: {closest_data_point.get('peak_15m', 'N/A'):.4f}, "
                  f"trough: {closest_data_point.get('trough_15m', 'N/A'):.4f}")

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
        major_peak_col = f'peak_{major_timeframe}'
        if major_peak_col in data_red.columns:
            fig.add_trace(go.Scatter(
                x=data_red.date,
                y=data_red[major_peak_col],
                name=f'Major Peak ({major_timeframe})',
                line=dict(color=px.colors.qualitative.Pastel[0], width=1)
            ), row=1, col=1)

        # Add Major Trough line
        major_trough_col = f'trough_{major_timeframe}'
        if major_trough_col in data_red.columns:
            fig.add_trace(go.Scatter(
                x=data_red.date,
                y=data_red[major_trough_col],
                name=f'Major Trough ({major_timeframe})',
                line=dict(color=px.colors.qualitative.Pastel[1], width=1)
            ), row=1, col=1)

        # Initialize strategy to get timeframes
        strategy = FractalStrategy(config=config)

        # Define timeframes for easier reference
        primary_timeframe = strategy.primary_timeframe
        major_timeframe = strategy.major_timeframe

        # Add Primary Peak line (from primary timeframe)
        primary_peak_col = f'peak_{primary_timeframe}'
        if primary_peak_col in data_red.columns:
            fig.add_trace(go.Scatter(
                x=data_red.date,
                y=data_red[primary_peak_col],
                name=f'Primary Peak ({primary_timeframe})',
                line=dict(color=px.colors.qualitative.Pastel[2], width=1)
            ), row=1, col=1)

        # Add Primary Trough line (from primary timeframe)
        primary_trough_col = f'trough_{primary_timeframe}'
        if primary_trough_col in data_red.columns:
            fig.add_trace(go.Scatter(
                x=data_red.date,
                y=data_red[primary_trough_col],
                name=f'Primary Trough ({primary_timeframe})',
                line=dict(color=px.colors.qualitative.Pastel[3], width=1)
            ), row=1, col=1)

        # Add markers for higher_high at major timeframe peaks
        # Only at the start of each hour for hourly timeframes
        major_peak_col = f'peak_{major_timeframe}'
        major_ha_upswing = f'ha_upswing_{major_timeframe}'
        major_ha_downswing = f'ha_downswing_{major_timeframe}'

        if (major_ha_upswing in data_red.columns and
            major_peak_col in data_red.columns and
            not data_red.empty):
            # Filter data to the first entry of each hour where the heikin ashi is up
            hh_data = data_red[data_red[major_ha_upswing] &
                    (data_red.index.minute == 0)]
            if not hh_data.empty:
                fig.add_trace(go.Scatter(
                    x=hh_data.date,
                    y=hh_data[major_peak_col],
                    mode='markers',
                    name=f'Up Swing ({major_timeframe})',
                    marker=dict(symbol='triangle-up', color=px.colors.qualitative.Pastel[4]),
                    hoverinfo='skip'
                ), row=1, col=1)

        # Add markers for lower_low at major timeframe troughs
        # Add markers for lower_low at major timeframe troughs, only at the start of each hour
        major_trough_col = f'trough_{major_timeframe}'

        if (major_ha_downswing in data_red.columns and
            major_trough_col in data_red.columns and
            not data_red.empty):
            # Filter data to the first entry of each hour where the heikin ashi is down
            ll_data = data_red[data_red[major_ha_downswing] &
                    (data_red.index.minute == 0)]
            if not ll_data.empty:
                fig.add_trace(go.Scatter(
                    x=ll_data.date,
                    y=ll_data[major_trough_col],
                    mode='markers',
                    name=f'Down Swing ({major_timeframe})',
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

        # Add Choppiness Index for primary timeframe
        primary_chop_col = f'chop_{primary_timeframe}'
        if primary_chop_col in data_red.columns:
            fig.add_trace(go.Scatter(
                x=data_red.date,
                y=data_red[primary_chop_col],
                name=f'Chop ({primary_timeframe})',
                line=dict(color='orange', width=1)
            ), row=3, col=1)

        # Add Choppiness Index for major timeframe
        major_chop_col = f'chop_{major_timeframe}'
        if major_chop_col in data_red.columns:
            fig.add_trace(go.Scatter(
                x=data_red.date,
                y=data_red[major_chop_col],
                name=f'Chop ({major_timeframe})',
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
        html_filename = f"{project_root}/user_data/chart_5m_{pair_symbol}_{start_date}_{end_date}.html"
        png_filename = f"{project_root}/user_data/chart_5m_{pair_symbol}.png"

        fig.write_html(html_filename)

            # Add navigation and asset selector
        with open(html_filename, 'r+') as f:
            content = f.read()
            body_index = content.find('<body>') + 6
            nav_html = f'''
            <div style="padding: 10px; background: #1f1f1f; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    {f'<a href="chart_5m_{pair_symbol}_{(datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")}_{start_date}.html" style="color: white; text-decoration: none; padding: 5px 10px; border: 1px solid #666; border-radius: 4px;">← Previous</a>'
                    if (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=3)) >= datetime.strptime(overall_start, "%Y-%m-%d")
                    else '<span style="color: #666; padding: 5px 10px;">← Start</span>'}
                    <span style="color: #888; margin: 0 15px;">{start_date} to {end_date}</span>
                    {f'<a href="chart_5m_{pair_symbol}_{end_date}_{(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")}.html" style="color: white; text-decoration: none; padding: 5px 10px; border: 1px solid #666; border-radius: 4px;">Next →</a>'
                    if (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=3)) <= datetime.strptime(overall_end, "%Y-%m-%d")
                    else '<span style="color: #666; padding: 5px 10px;">End →</span>'}
                </div>
                <select onchange="window.location.href=this.value.replace('ASSET',this.options[this.selectedIndex].text)"
                        style="padding: 5px; background: #333; color: white; border: 1px solid #666; border-radius: 4px;">
                    <option value="">Select Asset</option>
                    {''.join(f'<option value="chart_5m_ASSET_{start_date}_{end_date}.html">{sym}</option>' for sym in pairs_symbols)}
                </select>
            </div>
            '''
            modified_content = content[:body_index] + nav_html + content[body_index:]
            f.seek(0)
            f.write(modified_content)
            f.truncate()

        print(f"Interactive chart for {pair} has been saved to {html_filename}")

        # Save a static PNG version
        # fig.write_image(png_filename)
        # print(f"Static chart for {pair} has been saved to {png_filename}")
