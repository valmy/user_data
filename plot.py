import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import typer

from freqtrade.configuration import Configuration, TimeRange
from freqtrade.data.btanalysis import extract_trades_of_period, load_backtest_data
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import load_data
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import ExchangeResolver, StrategyResolver
from freqtrade.strategy import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Backtesting analysis script")


def load_backtest_trades(config: Configuration) -> list:
    filename = config.get("exportfilename")
    if not filename.is_dir() and not filename.is_file():
        logger.warning("Backtest file is missing skipping trades.")
        return []
    try:
        trades = load_backtest_data(filename, config.get("strategy"))
    except ValueError as e:
        raise OperationalException(e) from e
    return trades


@app.command()
def main(
    timerange: str = typer.Option(
        "20250801-",
        "--timerange",
        help="Period for analysis (YYYYMMDD-YYYYMMDD format)",
        show_default=True,
    ),
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Launch a Dash web app (dark theme) showing ETH/USDT:USDT data",
    ),
    port: int = typer.Option(8050, "--port", help="Port for the web server"),
) -> None:
    project_root = "."
    os.chdir(project_root)
    if not Path("user_data/config-backtest.json").is_file():
        logger.error("Wrong path, config-backtest.json not found")
        raise typer.Exit(code=1)

    # Set timerange to use
    tr = TimeRange.parse_timerange(timerange)

    # Initialize configuration object
    config = Configuration.from_files(["user_data/config-backtest.json"])

    # Initialize exchange/strategy and DP once
    exchange = ExchangeResolver.load_exchange(config)
    strategy = StrategyResolver.load_strategy(config)
    IStrategy.dp = DataProvider(config, exchange)
    strategy.ft_bot_start()
    strategy_safe_wrapper(strategy.bot_loop_start)(current_time=datetime.now(UTC))

    # Get available pairs
    whitelist = (
        set(config["exchange"]["pair_whitelist"])
        if "exchange" in config and "pair_whitelist" in config["exchange"]
        else set()
    )
    blacklist = set(config["exchange"].get("pair_blacklist", [])) if "exchange" in config else set()
    available_pairs = sorted(list(whitelist - blacklist))
    default_pair = (
        "ETH/USDT:USDT"
        if "ETH/USDT:USDT" in available_pairs
        else (available_pairs[0] if available_pairs else "ETH/USDT:USDT")
    )

    if serve:
        # Lazy import to allow normal CLI usage without dash dependencies
        try:
            import dash_mantine_components as dmc
            from dash import Dash, dcc
            from dash.dependencies import Input, Output, State
        except Exception:  # ImportError or other runtime import issues
            typer.echo(
                "dash and/or dash-mantine-components are not installed.\n"
                "Install with: pip install dash dash-mantine-components",
                err=True,
            )
            raise typer.Exit(code=1)

        try:
            import pandas as pd
            import plotly.graph_objects as go
        except Exception:
            typer.echo(
                "plotly and pandas are required.\nInstall with: pip install plotly pandas",
                err=True,
            )
            raise typer.Exit(code=1)

        # Determine timeframe
        tf = config.get("timeframe") or getattr(strategy, "timeframe", None)
        if not tf:
            tf = "3m"
            typer.echo(
                "Timeframe not found in config or strategy. Falling back to default '3m'.",
                err=True,
            )

        # Initial selected pair and data load
        selected_pair = default_pair
        data = load_data(
            datadir=config.get("datadir"),
            pairs=[selected_pair],
            timeframe=tf,
            timerange=tr,
            startup_candles=0,
            data_format=config["dataformat_ohlcv"],
            candle_type=config.get("candle_type_def"),
        )

        df = data.get(selected_pair)
        if df is None:
            typer.echo(f"{selected_pair} data not found in the loaded dataset.", err=True)
            raise typer.Exit(code=1)

        # Add indicators
        df_analyzed = strategy.analyze_ticker(df, {"pair": selected_pair})

        # Prepare df for plotting (preserve datetimes)
        try:
            df_plot = df_analyzed.copy()
            if df_plot.index.name and df_plot.index.name not in df_plot.columns:
                df_plot = df_plot.reset_index()
        except Exception:
            df_plot = df

        # Make dataframe readable for the table (stringify datetimes)
        try:
            df_reset = df_analyzed.copy()
            if df_reset.index.name and df_reset.index.name not in df_reset.columns:
                df_reset = df_reset.reset_index()
            for col in df_reset.columns:
                if str(df_reset[col].dtype).startswith("datetime"):
                    df_reset[col] = df_reset[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            df_reset = df

        # load trades
        trades = load_backtest_trades(config)

        # Build candlestick figure
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_plot["date"] if "date" in df_plot.columns else df_plot.index,
                    open=df_plot["open"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                    close=df_plot["close"],
                    increasing_line_color="lime",
                    increasing_fillcolor="lime",
                    decreasing_line_color="#ff4d4d",
                    decreasing_fillcolor="#ff4d4d",
                    name="ETH/USDT:USDT",
                )
            ]
        )
        # Set initial visible window to last ~8 hours (200 candles @ 3m)
        try:
            import pandas as pd

            x_values = df_plot["date"] if "date" in df_plot.columns else df_plot.index
            if len(x_values) >= 2:
                # Compute end as last timestamp, start as 200th from last (or first if shorter)
                end = x_values.iloc[-1]
                start_idx = max(0, len(x_values) - 200)
                start = x_values.iloc[start_idx]
                fig.update_xaxes(range=[start, end])

                # Compute initial Y range from only the visible window
                x_series = df_plot["date"] if "date" in df_plot.columns else df_plot.index
                x0_ts = pd.to_datetime(start)
                x1_ts = pd.to_datetime(end)
                mask = (pd.to_datetime(x_series) >= x0_ts) & (pd.to_datetime(x_series) <= x1_ts)
                if mask.any():
                    sub = df_plot.loc[mask]
                    ymin = float(sub["low"].min())
                    ymax = float(sub["high"].max())
                    if ymin < ymax:
                        pad = (ymax - ymin) * 0.05 or 1.0
                        fig.update_yaxes(range=[ymin - pad, ymax + pad])
        except Exception:
            # If anything goes wrong, fall back to default full range
            pass

        # Build pair selector dropdown
        pair_dropdown = dmc.Select(
            label="Pair",
            data=[{"value": p, "label": p} for p in available_pairs],
            value=selected_pair,
            searchable=True,
            clearable=False,
            size="sm",
            id="pair-select",
            style={"width": 280},
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(fixedrange=False),
            uirevision="ohlc-1",
        )

        # Build table header and rows (limit rows for performance)
        # Wrap chart in a responsive container
        chart = dmc.Paper(
            withBorder=True,
            # Store df used for plotting in app state via hidden component id
            # (not visible; used by callback to recompute y-range)
            # Using dmc.Paper solely to keep Mantine tree consistent.
            shadow="xs",
            p="sm",
            radius="sm",
            children=[
                dmc.Title("Candlestick", order=3),
                dmc.Space(h=8),
                dcc.Graph(id="ohlc-graph", figure=fig, style={"height": "80vh"}),
            ],
        )

        columns = [str(c) for c in df_reset.columns]
        max_rows = 200
        header = dmc.TableThead(dmc.TableTr([dmc.TableTh(col) for col in columns]))
        body_rows = []
        for row in df_reset.tail(max_rows).itertuples(index=False, name=None):
            body_rows.append(dmc.TableTr([dmc.TableTd("" if v is None else str(v)) for v in row]))
        body = dmc.TableTbody(body_rows)

        table = dmc.Table(
            id="data-table",
            children=[header, body],
            striped="odd",
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
            horizontalSpacing="sm",
            verticalSpacing="xs",
            style={"minWidth": 800},
        )

        # Build trades table
        if len(trades) > 0:
            # Select and format columns for display
            trades_df = trades[
                [
                    "pair",
                    "is_open",
                    "is_short",
                    "open_date",
                    "close_date",
                    "open_rate",
                    "close_rate",
                    "stake_amount",
                    "amount",
                    "profit_ratio",
                    "profit_abs",
                    "enter_tag",
                    "exit_reason",
                ]
            ]
            for col in ["open_date", "close_date"]:
                trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            trades_df = pd.DataFrame()

        if not trades_df.empty:
            trades_columns = [str(c) for c in trades_df.columns]
            trades_header = dmc.TableThead(
                dmc.TableTr([dmc.TableTh(col) for col in trades_columns])
            )
            trades_body_rows = [
                dmc.TableTr([dmc.TableTd("" if v is None else str(v)) for v in row])
                for row in trades_df.itertuples(index=False, name=None)
            ]
            trades_body = dmc.TableTbody(trades_body_rows)
            trades_table = dmc.Table(
                id="trades-table",
                children=[trades_header, trades_body],
                striped="odd",
                highlightOnHover=True,
                withTableBorder=True,
                withColumnBorders=True,
                horizontalSpacing="sm",
                verticalSpacing="xs",
                style={"minWidth": 800},
            )
        else:
            trades_table = dmc.Alert(
                title="No Trades",
                children="No trades found in backtest results.",
                color="blue",
                variant="filled",
            )

        # Create Dash app with dark theme
        dash_app = Dash(__name__)
        dash_app.title = "ETH/USDT:USDT Data"
        dash_app.layout = dmc.MantineProvider(
            forceColorScheme="dark",
            children=[
                dmc.Container(
                    size="xl",
                    px="md",
                    children=[
                        dmc.Space(h=16),
                        dmc.Title("OHLCV Viewer", order=2),
                        dmc.Space(h=8),
                        dmc.Group([pair_dropdown]),
                        dmc.Space(h=8),
                        chart,
                        dmc.Space(h=12),
                        dmc.Title("Price Data", order=4),
                        dmc.ScrollArea(
                            style={"height": "55vh"},
                            type="scroll",
                            children=table,
                        ),
                        dmc.Space(h=12),
                        dmc.Title("Trades", order=4),
                        dmc.ScrollArea(
                            style={"height": "55vh"},
                            type="scroll",
                            children=trades_table,
                        ),
                        dmc.Space(h=16),
                    ],
                ),
            ],
        )

        # Helpers to reduce duplication
        def _compute_visible_y_range(x_vals, lows, highs, x0, x1, pad_ratio=0.05):
            try:
                import pandas as pd

                x0_ts = pd.to_datetime(x0)
                x1_ts = pd.to_datetime(x1)
                ymin = None
                ymax = None
                for xv, lo, hi in zip(x_vals, lows, highs):
                    xv_ts = pd.to_datetime(xv)
                    if x0_ts <= xv_ts <= x1_ts:
                        ymin = float(lo) if ymin is None else min(ymin, float(lo))
                        ymax = float(hi) if ymax is None else max(ymax, float(hi))
                if ymin is None or ymax is None or not (ymin < ymax):
                    return None
                pad = (ymax - ymin) * pad_ratio or 1.0
                return [ymin - pad, ymax + pad]
            except Exception:
                return None

        def _build_table_children(df_table):
            columns2 = [str(c) for c in df_table.columns]
            header2 = dmc.TableThead(dmc.TableTr([dmc.TableTh(col) for col in columns2]))
            body_rows2 = [
                dmc.TableTr([dmc.TableTd("" if v is None else str(v)) for v in row])
                for row in df_table.tail(200).itertuples(index=False, name=None)
            ]
            body2 = dmc.TableTbody(body_rows2)
            return [header2, body2]

        @dash_app.callback(
            Output("ohlc-graph", "figure"),
            Output("data-table", "children"),
            Input("pair-select", "value"),
            Input("ohlc-graph", "relayoutData"),
            State("ohlc-graph", "figure"),
            State("data-table", "children"),
        )
        def _on_pair_or_view_change(pair_value, relayout_data, current_fig, table_children):  # type: ignore
            try:
                import dash
                import pandas as pd

                ctx = dash.callback_context  # for trigger inspection
                triggered = ctx.triggered[0]["prop_id"] if ctx and ctx.triggered else ""
                logger.info(f"pair/view callback triggered={triggered} pair={pair_value}")

                # 1) If relayout changed x-range and no pair change, just rescale y
                if (
                    triggered.endswith("relayoutData")
                    and current_fig
                    and current_fig.get("data")
                    and relayout_data
                ):
                    x0 = x1 = None
                    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
                        x0 = relayout_data["xaxis.range[0]"]
                        x1 = relayout_data["xaxis.range[1]"]
                    elif "xaxis.range" in relayout_data and isinstance(
                        relayout_data["xaxis.range"], list
                    ):
                        x0, x1 = relayout_data["xaxis.range"]
                    elif relayout_data.get("xaxis.autorange"):
                        current_fig["layout"]["yaxis"].pop("range", None)
                        current_fig["layout"]["yaxis"]["autorange"] = True
                        return current_fig, table_children

                    if x0 is not None and x1 is not None:
                        trace = current_fig["data"][0]
                        x_vals = trace.get("x", [])
                        lows = trace.get("low", [])
                        highs = trace.get("high", [])
                        vrange = _compute_visible_y_range(x_vals, lows, highs, x0, x1)
                        if vrange:
                            current_fig["layout"]["yaxis"]["range"] = vrange
                            current_fig["layout"]["yaxis"]["autorange"] = False
                        return current_fig, table_children

                # 2) Pair changed - reload data and rebuild fig + table
                new_data = load_data(
                    datadir=config.get("datadir"),
                    pairs=[pair_value],
                    timeframe=tf,
                    timerange=tr,
                    startup_candles=0,
                    data_format=config["dataformat_ohlcv"],
                    candle_type=config.get("candle_type_def"),
                )
                new_df = new_data.get(pair_value)
                if new_df is None or new_df.empty:
                    return current_fig, dmc.Alert(
                        title="No data",
                        children=f"No OHLCV data available for {pair_value}.",
                        color="red",
                        variant="filled",
                    )

                # Add indicators
                df_analyzed = strategy.analyze_ticker(new_df, {"pair": pair_value})

                df_plot2 = df_analyzed.copy()
                if df_plot2.index.name and df_plot2.index.name not in df_plot2.columns:
                    df_plot2 = df_plot2.reset_index()

                fig2 = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df_plot2["date"] if "date" in df_plot2.columns else df_plot2.index,
                            open=df_plot2["open"],
                            high=df_plot2["high"],
                            low=df_plot2["low"],
                            close=df_plot2["close"],
                            increasing_line_color="lime",
                            increasing_fillcolor="lime",
                            decreasing_line_color="#ff4d4d",
                            decreasing_fillcolor="#ff4d4d",
                            name=pair_value,
                        )
                    ]
                )

                # Initial range: last ~8 hours and y from visible
                x_vals2 = df_plot2["date"] if "date" in df_plot2.columns else df_plot2.index
                if len(x_vals2) >= 2:
                    end2 = x_vals2.iloc[-1]
                    start2 = x_vals2.iloc[max(0, len(x_vals2) - 200)]
                    fig2.update_xaxes(range=[start2, end2])
                    vrange2 = _compute_visible_y_range(
                        x_vals2,
                        df_plot2["low"],
                        df_plot2["high"],
                        start2,
                        end2,
                    )
                    if vrange2:
                        fig2.update_yaxes(range=vrange2)

                fig2.update_layout(
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Price (USDT)",
                    xaxis=dict(rangeslider=dict(visible=True)),
                    yaxis=dict(fixedrange=False),
                    uirevision="ohlc-1",
                )

                # Build table children
                df_table = df_analyzed.copy()
                if df_table.index.name and df_table.index.name not in df_table.columns:
                    df_table = df_table.reset_index()
                for col in df_table.columns:
                    if str(df_table[col].dtype).startswith("datetime"):
                        df_table[col] = df_table[col].dt.strftime("%Y-%m-%d %H:%M:%S")
                return fig2, _build_table_children(df_table)
            except Exception as e:
                return current_fig, dmc.Alert(
                    title="Error", children=str(e), color="red", variant="filled"
                )

        # Deprecated: y-rescale handled by combined callback above
        # (kept here for readability, but not registered)

        # Run server on localhost
        dash_app.run(host="127.0.0.1", port=port, debug=False)
        return

    # Normal (non-serve) execution
    markets = exchange.get_markets().keys()
    if "pairs" in config:
        pairs = expand_pairlist(config["pairs"], markets)
    else:
        pairs = expand_pairlist(config["exchange"]["pair_whitelist"], markets)

    data = load_data(
        datadir=config.get("datadir"),
        pairs=pairs,
        timeframe=config["timeframe"],
        timerange=tr,
        startup_candles=0,
        data_format=config["dataformat_ohlcv"],
        candle_type=config.get("candle_type_def"),
    )

    # Add indicators
    selected_pair = "ETH/USDT:USDT"
    df_analyzed = strategy.analyze_ticker(data[selected_pair], {"pair": selected_pair})

    print(df_analyzed)
    print(df_analyzed.columns)

    trades = load_backtest_trades(config)
    print(trades.columns)


if __name__ == "__main__":
    app()
