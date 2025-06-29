#!/usr/bin/env python3
"""
Freqtrade MCP Server

A Model Context Protocol server that provides access to freqtrade pair data and backtesting results.
This server exposes freqtrade data through MCP resources and tools for analysis.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Add freqtrade to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from freqtrade.configuration import Configuration
from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.data.dataprovider import DataProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Freqtrade Data Server")

# Global configuration and data cache
_config: Optional[Dict] = None
_backtest_stats: Optional[Dict] = None
_backtest_trades: Optional[pd.DataFrame] = None
_strategy = None
_exchange = None

def get_config() -> Dict:
    """Get or initialize freqtrade configuration."""
    global _config
    if _config is None:
        try:
            config_path = Path("user_data/config.json")
            if config_path.exists():
                _config = Configuration.from_files([str(config_path)])
            else:
                # Fallback configuration
                _config = {
                    "datadir": Path("user_data/data"),
                    "timeframe": "5m",
                    "strategy": "FractalStrategy",
                    "user_data_dir": Path("user_data"),
                }
            logger.info(f"Loaded configuration with datadir: {_config.get('datadir')}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            _config = {
                "datadir": Path("user_data/data"),
                "timeframe": "5m",
                "strategy": "FractalStrategy",
                "user_data_dir": Path("user_data"),
            }
    return _config

def get_backtest_data() -> tuple[Optional[Dict], Optional[pd.DataFrame]]:
    """Get or load backtest statistics and trades data."""
    global _backtest_stats, _backtest_trades

    if _backtest_stats is None or _backtest_trades is None:
        try:
            config = get_config()
            backtest_dir = config["user_data_dir"] / "backtest_results"

            if backtest_dir.exists():
                _backtest_stats = load_backtest_stats(backtest_dir)
                _backtest_trades = load_backtest_data(backtest_dir)
                logger.info(f"Loaded backtest data with {len(_backtest_trades) if _backtest_trades is not None else 0} trades")
            else:
                logger.warning(f"Backtest directory not found: {backtest_dir}")
                _backtest_stats = {}
                _backtest_trades = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load backtest data: {e}")
            _backtest_stats = {}
            _backtest_trades = pd.DataFrame()

    return _backtest_stats, _backtest_trades

def get_strategy():
    """Get or initialize strategy."""
    global _strategy, _exchange

    if _strategy is None:
        try:
            config = get_config()

            # Try to load strategy with proper configuration
            from freqtrade.resolvers import StrategyResolver
            from freqtrade.exchange import Exchange

            # Ensure strategy is in config
            if 'strategy' not in config or not config['strategy']:
                # Try to find a strategy file
                strategy_dir = Path(config.get('user_data_dir', 'user_data')) / 'strategies'
                if strategy_dir.exists():
                    strategy_files = list(strategy_dir.glob('*.py'))
                    if strategy_files:
                        # Use FractalStrategy if available, otherwise first strategy
                        fractal_strategy = strategy_dir / 'FractalStrategy.py'
                        if fractal_strategy.exists():
                            config['strategy'] = 'FractalStrategy'
                        else:
                            # Use first available strategy
                            strategy_name = strategy_files[0].stem
                            config['strategy'] = strategy_name
                        logger.info(f"Auto-selected strategy: {config['strategy']}")
                    else:
                        raise Exception("No strategy files found in strategies directory")
                else:
                    raise Exception("Strategies directory not found")

            # Load strategy using the same pattern as run.py
            _strategy = StrategyResolver.load_strategy(config)

            # Initialize exchange and DataProvider following run.py pattern
            if _exchange is None:
                try:
                    # Use Binance exchange like in run.py
                    from freqtrade.exchange.binance import Binance
                    _exchange = Binance(config)
                    _strategy.dp = DataProvider(config, _exchange, None)
                    _strategy.ft_bot_start()

                    logger.info(f"Loaded strategy: {config.get('strategy')} with Binance exchange")
                except Exception as exchange_error:
                    logger.warning(f"Binance exchange initialization failed: {exchange_error}")
                    # Try with generic Exchange as fallback
                    try:
                        from freqtrade.exchange import Exchange
                        _exchange = Exchange(config)
                        _strategy.dp = DataProvider(config, _exchange, None)
                        _strategy.ft_bot_start()
                        logger.info(f"Loaded strategy: {config.get('strategy')} with generic exchange")
                    except Exception as fallback_error:
                        logger.error(f"All exchange initialization failed: {fallback_error}")
                        _strategy = None

        except Exception as e:
            logger.error(f"Failed to load strategy: {e}")
            # Try to provide a fallback or more detailed error
            try:
                import traceback
                logger.error(f"Strategy loading traceback: {traceback.format_exc()}")
            except:
                pass
            _strategy = None

    return _strategy

def analyze_with_strategy(strategy, candles, pair):
    """Helper function to analyze candles with strategy, following run.py pattern."""
    # Follow the exact same pattern as run.py
    metadata = {"pair": pair}
    return strategy.analyze_ticker(candles, metadata)

# Pydantic models for structured responses
class TradeInfo(BaseModel):
    """Information about a single trade."""
    pair: str
    open_date: str
    close_date: str
    profit_ratio: float
    profit_abs: float
    exit_reason: str
    trade_duration: str
    is_short: bool = False
    open_rate: float
    close_rate: float
    amount: float
    stake_amount: float

class PairStats(BaseModel):
    """Statistics for a trading pair."""
    pair: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    avg_profit: float
    max_profit: float
    min_profit: float

class BacktestSummary(BaseModel):
    """Summary of backtest results."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    start_date: str
    end_date: str
    pairs: List[str]

# MCP Resources
@mcp.resource("freqtrade://config")
def get_freqtrade_config() -> str:
    """Get the current freqtrade configuration."""
    config = get_config()
    # Remove sensitive information
    safe_config = {k: str(v) for k, v in config.items()
                   if not any(sensitive in k.lower() for sensitive in ['key', 'secret', 'password', 'token'])}
    return json.dumps(safe_config, indent=2, default=str)

@mcp.resource("freqtrade://backtest/summary")
def get_backtest_summary() -> str:
    """Get a summary of the latest backtest results."""
    stats, trades = get_backtest_data()

    if not stats or trades is None or trades.empty:
        return json.dumps({"error": "No backtest data available"})

    try:
        # Get strategy name from stats
        strategy_names = list(stats.get("strategy", {}).keys())
        if not strategy_names:
            return json.dumps({"error": "No strategy data found"})

        strategy_name = strategy_names[0]
        strategy_stats = stats["strategy"][strategy_name]

        summary = BacktestSummary(
            strategy_name=strategy_name,
            total_trades=len(trades),
            winning_trades=len(trades[trades["profit_ratio"] > 0]),
            losing_trades=len(trades[trades["profit_ratio"] <= 0]),
            win_rate=len(trades[trades["profit_ratio"] > 0]) / len(trades) * 100 if len(trades) > 0 else 0,
            total_profit=trades["profit_abs"].sum() if not trades.empty else 0,
            max_drawdown=strategy_stats.get("max_drawdown_abs", 0),
            start_date=str(trades["open_date"].min()) if not trades.empty else "",
            end_date=str(trades["close_date"].max()) if not trades.empty else "",
            pairs=trades["pair"].unique().tolist() if not trades.empty else []
        )

        return summary.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error creating backtest summary: {e}")
        return json.dumps({"error": f"Failed to create summary: {str(e)}"})

@mcp.resource("freqtrade://backtest/trades")
def get_all_trades() -> str:
    """Get all trades from the latest backtest."""
    _, trades = get_backtest_data()

    if trades is None or trades.empty:
        return json.dumps({"error": "No trade data available"})

    # Convert trades to list of TradeInfo objects
    trade_list = []
    for _, trade in trades.iterrows():
        trade_info = TradeInfo(
            pair=trade["pair"],
            open_date=str(trade["open_date"]),
            close_date=str(trade["close_date"]),
            profit_ratio=float(trade["profit_ratio"]),
            profit_abs=float(trade["profit_abs"]),
            exit_reason=trade["exit_reason"],
            trade_duration=str(trade["trade_duration"]),
            is_short=bool(trade.get("is_short", False)),
            open_rate=float(trade["open_rate"]),
            close_rate=float(trade["close_rate"]),
            amount=float(trade["amount"]),
            stake_amount=float(trade["stake_amount"])
        )
        trade_list.append(trade_info.model_dump())

    return json.dumps(trade_list, indent=2)

@mcp.resource("freqtrade://pairs/list")
def get_available_pairs() -> str:
    """Get list of available trading pairs."""
    _, trades = get_backtest_data()
    config = get_config()

    pairs_info = {
        "backtest_pairs": [],
        "data_directory": str(config.get("datadir", "")),
        "timeframe": config.get("timeframe", "5m")
    }

    if trades is not None and not trades.empty:
        pairs_info["backtest_pairs"] = sorted(trades["pair"].unique().tolist())

    # Try to get pairs from data directory
    try:
        data_dir = Path(config.get("datadir", "user_data/data"))
        if data_dir.exists():
            # Look for feather files
            feather_files = list(data_dir.glob("*.feather"))
            available_pairs = []
            for file in feather_files:
                # Extract pair name from filename (e.g., BTC_USDT-5m.feather)
                pair_name = file.stem.split("-")[0].replace("_", "/")
                if ":" not in pair_name:
                    pair_name += ":USDT"  # Add quote currency if missing
                available_pairs.append(pair_name)
            pairs_info["available_data_pairs"] = sorted(set(available_pairs))
    except Exception as e:
        logger.error(f"Error scanning data directory: {e}")
        pairs_info["available_data_pairs"] = []

    return json.dumps(pairs_info, indent=2)

@mcp.resource("freqtrade://pair/{pair}/stats")
def get_pair_statistics(pair: str) -> str:
    """Get statistics for a specific trading pair."""
    _, trades = get_backtest_data()

    if trades is None or trades.empty:
        return json.dumps({"error": "No trade data available"})

    # Filter trades for the specific pair
    pair_trades = trades[trades["pair"] == pair]

    if pair_trades.empty:
        return json.dumps({"error": f"No trades found for pair {pair}"})

    try:
        stats = PairStats(
            pair=pair,
            total_trades=len(pair_trades),
            winning_trades=len(pair_trades[pair_trades["profit_ratio"] > 0]),
            losing_trades=len(pair_trades[pair_trades["profit_ratio"] <= 0]),
            win_rate=len(pair_trades[pair_trades["profit_ratio"] > 0]) / len(pair_trades) * 100,
            total_profit=float(pair_trades["profit_abs"].sum()),
            avg_profit=float(pair_trades["profit_abs"].mean()),
            max_profit=float(pair_trades["profit_abs"].max()),
            min_profit=float(pair_trades["profit_abs"].min())
        )

        return stats.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error calculating pair statistics: {e}")
        return json.dumps({"error": f"Failed to calculate statistics: {str(e)}"})

# MCP Tools
@mcp.tool()
def get_pair_data(pair: str, timeframe: str = "5m", limit: int = 1000) -> str:
    """
    Load historical OHLCV data for a specific trading pair.

    Args:
        pair: Trading pair (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe for the data (default: '5m')
        limit: Maximum number of candles to return (default: 1000)
    """
    try:
        config = get_config()
        data_location = config.get("datadir", Path("user_data/data"))

        # Load pair history
        candles = load_pair_history(
            datadir=data_location,
            timeframe=timeframe,
            pair=pair,
            data_format="feather",
            candle_type=CandleType.FUTURES,
        )

        if candles.empty:
            return json.dumps({"error": f"No data found for {pair} with timeframe {timeframe}"})

        # Limit the data if requested
        if limit > 0:
            candles = candles.tail(limit)

        # Convert to JSON-serializable format
        result = {
            "pair": pair,
            "timeframe": timeframe,
            "data_points": len(candles),
            "start_date": str(candles.index[0]) if not candles.empty else None,
            "end_date": str(candles.index[-1]) if not candles.empty else None,
            "data": candles.reset_index().to_dict(orient="records")
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error loading pair data: {e}")
        return json.dumps({"error": f"Failed to load data: {str(e)}"})

@mcp.tool()
def analyze_trades_by_exit_reason(pair: Optional[str] = None) -> str:
    """
    Analyze trades grouped by exit reason.

    Args:
        pair: Optional trading pair to filter by
    """
    try:
        _, trades = get_backtest_data()

        if trades is None or trades.empty:
            return json.dumps({"error": "No trade data available"})

        # Filter by pair if specified
        if pair:
            trades = trades[trades["pair"] == pair]
            if trades.empty:
                return json.dumps({"error": f"No trades found for pair {pair}"})

        # Group by exit reason
        exit_analysis = trades.groupby("exit_reason").agg({
            "profit_ratio": ["count", "mean", "sum"],
            "profit_abs": ["sum", "mean"],
            "trade_duration": "mean"
        }).round(4)

        # Flatten column names
        exit_analysis.columns = ["_".join(col).strip() for col in exit_analysis.columns]

        # Convert to dictionary
        result = {
            "analysis_scope": f"All pairs" if not pair else f"Pair: {pair}",
            "total_trades": len(trades),
            "exit_reasons": exit_analysis.to_dict(orient="index")
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error analyzing trades by exit reason: {e}")
        return json.dumps({"error": f"Failed to analyze trades: {str(e)}"})

@mcp.tool()
def get_strategy_signals(pair: str, timeframe: str = "5m", limit: int = 100) -> str:
    """
    Generate trading signals for a pair using the loaded strategy.

    Args:
        pair: Trading pair (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe for analysis (default: '5m')
        limit: Number of recent candles to analyze (default: 100)
    """
    try:
        config = get_config()
        strategy = get_strategy()

        if strategy is None:
            return json.dumps({"error": "Strategy not loaded"})

        # Load pair data
        data_location = config.get("datadir", Path("user_data/data"))
        candles = load_pair_history(
            datadir=data_location,
            timeframe=timeframe,
            pair=pair,
            data_format="feather",
            candle_type=CandleType.FUTURES,
        )

        if candles.empty:
            return json.dumps({"error": f"No data found for {pair}"})

        # Limit data
        if limit > 0:
            candles = candles.tail(limit)

        # Generate signals using strategy
        try:
            df = analyze_with_strategy(strategy, candles, pair)
        except Exception as e:
            return json.dumps({"error": f"Strategy analysis failed: {str(e)}"})

        # Extract signal information
        signals = []
        for idx, row in df.iterrows():
            signal_info = {
                "date": str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "enter_long": bool(row.get("enter_long", False)),
                "enter_short": bool(row.get("enter_short", False)),
                "exit_long": bool(row.get("exit_long", False)),
                "exit_short": bool(row.get("exit_short", False)),
            }

            # Add strategy-specific indicators if available
            for col in df.columns:
                if col not in ["open", "high", "low", "close", "volume", "date",
                              "enter_long", "enter_short", "exit_long", "exit_short"]:
                    try:
                        signal_info[col] = float(row[col]) if pd.notna(row[col]) else None
                    except (ValueError, TypeError):
                        signal_info[col] = str(row[col]) if pd.notna(row[col]) else None

            signals.append(signal_info)

        # Count signals
        enter_long_count = sum(1 for s in signals if s["enter_long"])
        enter_short_count = sum(1 for s in signals if s["enter_short"])

        result = {
            "pair": pair,
            "timeframe": timeframe,
            "strategy": config.get("strategy", "Unknown"),
            "data_points": len(signals),
            "enter_long_signals": enter_long_count,
            "enter_short_signals": enter_short_count,
            "signals": signals
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error generating strategy signals: {e}")
        return json.dumps({"error": f"Failed to generate signals: {str(e)}"})

@mcp.tool()
def analyze_indicator_states(pair: str, timeframe: str = "5m", limit: int = 100, focus_on_signals: bool = True) -> str:
    """
    Analyze the states of indicators from strategy.analyze_ticker to understand trade decision factors.

    Args:
        pair: Trading pair (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe for analysis (default: '5m')
        limit: Number of recent candles to analyze (default: 100)
        focus_on_signals: If True, focus analysis on periods with entry/exit signals
    """
    try:
        config = get_config()
        strategy = get_strategy()

        if strategy is None:
            return json.dumps({"error": "Strategy not loaded"})

        # Load pair data
        data_location = config.get("datadir", Path("user_data/data"))
        candles = load_pair_history(
            datadir=data_location,
            timeframe=timeframe,
            pair=pair,
            data_format="feather",
            candle_type=CandleType.FUTURES,
        )

        if candles.empty:
            return json.dumps({"error": f"No data found for {pair}"})

        # Limit data
        if limit > 0:
            candles = candles.tail(limit)

        # Generate signals and indicators using strategy
        try:
            df = analyze_with_strategy(strategy, candles, pair)
        except Exception as e:
            return json.dumps({"error": f"Strategy analysis failed: {str(e)}"})

        # Identify indicator columns (exclude OHLCV and signal columns)
        base_columns = ["open", "high", "low", "close", "volume", "date"]
        signal_columns = ["enter_long", "enter_short", "exit_long", "exit_short"]
        indicator_columns = [col for col in df.columns
                           if col not in base_columns + signal_columns]

        # Analyze signal periods if requested
        signal_analysis = {}
        if focus_on_signals:
            # Find rows with signals - use proper boolean indexing to avoid numpy boolean subtraction issues
            enter_long_mask = df["enter_long"].astype(bool) if "enter_long" in df.columns else pd.Series(False, index=df.index)
            enter_short_mask = df["enter_short"].astype(bool) if "enter_short" in df.columns else pd.Series(False, index=df.index)
            exit_long_mask = df["exit_long"].astype(bool) if "exit_long" in df.columns else pd.Series(False, index=df.index)
            exit_short_mask = df["exit_short"].astype(bool) if "exit_short" in df.columns else pd.Series(False, index=df.index)

            entry_signals = df[enter_long_mask | enter_short_mask]
            exit_signals = df[exit_long_mask | exit_short_mask]

            if not entry_signals.empty:
                # Count signals by filtering the original dataframe
                long_entry_count = len(df[enter_long_mask])
                short_entry_count = len(df[enter_short_mask])

                signal_analysis["entry_signals"] = {
                    "count": len(entry_signals),
                    "long_entries": long_entry_count,
                    "short_entries": short_entry_count,
                    "indicator_states_at_entry": {}
                }

                # Analyze indicator states at entry signals
                for indicator in indicator_columns:
                    if indicator in entry_signals.columns:
                        values = entry_signals[indicator].dropna()
                        if not values.empty:
                            # Convert sample values to JSON-serializable format
                            sample_values = []
                            try:
                                sample_values = [float(x) if pd.api.types.is_numeric_dtype(values) else str(x)
                                               for x in values.tail(3).values if pd.notna(x)]
                            except (ValueError, TypeError):
                                sample_values = [str(x) for x in values.tail(3).values if pd.notna(x)]

                            signal_analysis["entry_signals"]["indicator_states_at_entry"][indicator] = {
                                "mean": float(values.mean()) if pd.api.types.is_numeric_dtype(values) else None,
                                "min": float(values.min()) if pd.api.types.is_numeric_dtype(values) else None,
                                "max": float(values.max()) if pd.api.types.is_numeric_dtype(values) else None,
                                "std": float(values.std()) if pd.api.types.is_numeric_dtype(values) else None,
                                "sample_values": sample_values
                            }

            if not exit_signals.empty:
                # Count signals by filtering the original dataframe
                long_exit_count = len(df[exit_long_mask])
                short_exit_count = len(df[exit_short_mask])

                signal_analysis["exit_signals"] = {
                    "count": len(exit_signals),
                    "long_exits": long_exit_count,
                    "short_exits": short_exit_count,
                    "indicator_states_at_exit": {}
                }

                # Analyze indicator states at exit signals
                for indicator in indicator_columns:
                    if indicator in exit_signals.columns:
                        values = exit_signals[indicator].dropna()
                        if not values.empty:
                            # Convert sample values to JSON-serializable format
                            sample_values = []
                            try:
                                sample_values = [float(x) if pd.api.types.is_numeric_dtype(values) else str(x)
                                               for x in values.tail(3).values if pd.notna(x)]
                            except (ValueError, TypeError):
                                sample_values = [str(x) for x in values.tail(3).values if pd.notna(x)]

                            signal_analysis["exit_signals"]["indicator_states_at_exit"][indicator] = {
                                "mean": float(values.mean()) if pd.api.types.is_numeric_dtype(values) else None,
                                "min": float(values.min()) if pd.api.types.is_numeric_dtype(values) else None,
                                "max": float(values.max()) if pd.api.types.is_numeric_dtype(values) else None,
                                "std": float(values.std()) if pd.api.types.is_numeric_dtype(values) else None,
                                "sample_values": sample_values
                            }

        # Overall indicator analysis
        indicator_analysis = {}
        for indicator in indicator_columns:
            if indicator in df.columns:
                values = df[indicator].dropna()
                if not values.empty:
                    analysis = {
                        "data_points": len(values),
                        "data_type": str(values.dtype),
                        "has_nulls": df[indicator].isnull().sum(),
                        "null_percentage": round(df[indicator].isnull().sum() / len(df) * 100, 2)
                    }

                    if pd.api.types.is_numeric_dtype(values):
                        # Calculate trend direction safely
                        trend_direction = "unknown"
                        if len(values) >= 10:
                            try:
                                latest_val = float(values.iloc[-1])
                                earlier_val = float(values.iloc[-10])
                                trend_direction = "up" if latest_val > earlier_val else "down"
                            except (ValueError, TypeError, IndexError):
                                trend_direction = "unknown"

                        # Calculate range safely to avoid boolean subtraction
                        min_val = float(values.min())
                        max_val = float(values.max())
                        range_val = max_val - min_val

                        analysis.update({
                            "mean": float(values.mean()),
                            "median": float(values.median()),
                            "std": float(values.std()),
                            "min": min_val,
                            "max": max_val,
                            "range": range_val,
                            "latest_value": float(values.iloc[-1]),
                            "trend_direction": trend_direction
                        })
                    else:
                        # For non-numeric indicators (boolean, categorical)
                        value_counts = values.value_counts()
                        # Convert value_counts to dict with string keys to avoid JSON serialization issues
                        value_distribution = {str(k): int(v) for k, v in value_counts.items()}

                        analysis.update({
                            "unique_values": len(value_counts),
                            "value_distribution": value_distribution,
                            "latest_value": str(values.iloc[-1]),
                            "most_common": str(value_counts.index[0]) if not value_counts.empty else None
                        })

                    indicator_analysis[indicator] = analysis

        result = {
            "pair": pair,
            "timeframe": timeframe,
            "strategy": config.get("strategy", "Unknown"),
            "analysis_period": {
                "start": str(df.index[0]) if not df.empty else None,
                "end": str(df.index[-1]) if not df.empty else None,
                "data_points": len(df)
            },
            "indicators_found": len(indicator_columns),
            "indicator_list": indicator_columns,
            "indicator_analysis": indicator_analysis,
            "signal_analysis": signal_analysis if focus_on_signals else {}
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error analyzing indicator states: {e}")
        return json.dumps({"error": f"Failed to analyze indicators: {str(e)}"})

@mcp.tool()
def analyze_trade_decision_factors(pair: str, timeframe: str = "5m", lookback_periods: int = 5) -> str:
    """
    Analyze the factors (indicators) that led to trade decisions by examining indicator states
    before and during signal generation.


    Args:
        pair: Trading pair (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe for analysis (default: '5m')
        lookback_periods: Number of periods to look back from signals for analysis
    """
    try:
        # Add timeout protection to prevent hanging
        import signal

        def timeout_handler(signum, frame):  # noqa: ARG001
            raise TimeoutError("Analysis timed out after 30 seconds")

        # Set a 30-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        config = get_config()
        strategy = get_strategy()

        if strategy is None:
            return json.dumps({"error": "Strategy not loaded"})

        # Load pair data
        data_location = config.get("datadir", Path("user_data/data"))
        candles = load_pair_history(
            datadir=data_location,
            timeframe=timeframe,
            pair=pair,
            data_format="feather",
            candle_type=CandleType.FUTURES,
        )

        if candles.empty:
            return json.dumps({"error": f"No data found for {pair}"})

        # Generate signals and indicators using strategy
        try:
            df = analyze_with_strategy(strategy, candles, pair)
        except Exception as e:
            return json.dumps({"error": f"Strategy analysis failed: {str(e)}"})

        # Identify indicator columns
        base_columns = ["open", "high", "low", "close", "volume", "date"]
        signal_columns = ["enter_long", "enter_short", "exit_long", "exit_short"]
        indicator_columns = [col for col in df.columns
                           if col not in base_columns + signal_columns]

        # Find signal points - use .astype(bool) to avoid numpy boolean subtraction issues
        entry_long_signals = df[df.get("enter_long", False).astype(bool) if "enter_long" in df.columns else pd.Series(False, index=df.index)].index
        entry_short_signals = df[df.get("enter_short", False).astype(bool) if "enter_short" in df.columns else pd.Series(False, index=df.index)].index
        exit_long_signals = df[df.get("exit_long", False).astype(bool) if "exit_long" in df.columns else pd.Series(False, index=df.index)].index
        exit_short_signals = df[df.get("exit_short", False).astype(bool) if "exit_short" in df.columns else pd.Series(False, index=df.index)].index

        def analyze_signal_context(signal_indices, signal_type):
            """Analyze indicator context around signals."""
            if len(signal_indices) == 0:
                return {"count": 0, "signal_type": signal_type, "analysis": {}}

            context_analysis = {
                "count": len(signal_indices),
                "signal_dates": [str(idx) for idx in signal_indices],
                "indicator_patterns": {},
                "price_context": {}
            }

            for signal_idx in signal_indices:
                # Get the position in the dataframe
                signal_pos = df.index.get_loc(signal_idx)

                # Define the analysis window (lookback periods before signal)
                start_pos = max(0, signal_pos - lookback_periods)
                end_pos = min(len(df), signal_pos + 1)

                window_df = df.iloc[start_pos:end_pos]

                # Analyze price context
                if signal_pos > 0:
                    price_change = (df.iloc[signal_pos]["close"] - df.iloc[signal_pos-1]["close"]) / df.iloc[signal_pos-1]["close"] * 100
                    context_analysis["price_context"][str(signal_idx)] = {
                        "price_at_signal": float(df.iloc[signal_pos]["close"]),
                        "price_change_percent": round(price_change, 4),
                        "volume_at_signal": float(df.iloc[signal_pos]["volume"])
                    }

                # Analyze each indicator's behavior leading to the signal
                for indicator in indicator_columns:
                    if indicator not in context_analysis["indicator_patterns"]:
                        context_analysis["indicator_patterns"][indicator] = {
                            "values_at_signals": [],
                            "trends_before_signals": [],
                            "volatility_before_signals": []
                        }

                    if indicator in window_df.columns:
                        indicator_values = window_df[indicator].dropna()

                        if not indicator_values.empty and pd.api.types.is_numeric_dtype(indicator_values):
                            # Value at signal
                            signal_value = float(df.iloc[signal_pos][indicator]) if pd.notna(df.iloc[signal_pos][indicator]) else None
                            context_analysis["indicator_patterns"][indicator]["values_at_signals"].append(signal_value)

                            # Trend analysis (slope of indicator before signal)
                            if len(indicator_values) >= 2:
                                trend = "up" if indicator_values.iloc[-1] > indicator_values.iloc[0] else "down"
                                context_analysis["indicator_patterns"][indicator]["trends_before_signals"].append(trend)

                                # Volatility (standard deviation of recent values)
                                volatility = float(indicator_values.std()) if len(indicator_values) > 1 else 0
                                context_analysis["indicator_patterns"][indicator]["volatility_before_signals"].append(volatility)

            # Summarize patterns
            for indicator in context_analysis["indicator_patterns"]:
                patterns = context_analysis["indicator_patterns"][indicator]

                # Calculate summary statistics
                values = [v for v in patterns["values_at_signals"] if v is not None]
                if values:
                    patterns["summary"] = {
                        "avg_value_at_signal": round(sum(values) / len(values), 4),
                        "min_value_at_signal": round(min(values), 4),
                        "max_value_at_signal": round(max(values), 4),
                        "value_range": round(max(values) - min(values), 4)
                    }

                # Trend patterns
                trends = patterns["trends_before_signals"]
                if trends:
                    trend_counts = {trend: trends.count(trend) for trend in set(trends)}
                    patterns["trend_summary"] = {
                        "most_common_trend": max(trend_counts, key=trend_counts.get),
                        "trend_distribution": trend_counts
                    }

                # Volatility patterns
                volatilities = patterns["volatility_before_signals"]
                if volatilities:
                    patterns["volatility_summary"] = {
                        "avg_volatility": round(sum(volatilities) / len(volatilities), 4),
                        "high_volatility_signals": sum(1 for v in volatilities if v > sum(volatilities) / len(volatilities))
                    }

            return context_analysis

        # Analyze each signal type
        analysis_results = {
            "pair": pair,
            "timeframe": timeframe,
            "strategy": config.get("strategy", "Unknown"),
            "lookback_periods": lookback_periods,
            "total_indicators": len(indicator_columns),
            "indicators_analyzed": indicator_columns,
            "signal_analysis": {
                "entry_long": analyze_signal_context(entry_long_signals, "entry_long"),
                "entry_short": analyze_signal_context(entry_short_signals, "entry_short"),
                "exit_long": analyze_signal_context(exit_long_signals, "exit_long"),
                "exit_short": analyze_signal_context(exit_short_signals, "exit_short")
            }
        }

        # Add correlation analysis between indicators and signals
        correlation_analysis = {}
        for indicator in indicator_columns:
            if indicator in df.columns and pd.api.types.is_numeric_dtype(df[indicator]):
                indicator_data = df[indicator].dropna()

                correlations = {}
                for signal_col in signal_columns:
                    if signal_col in df.columns:
                        # Convert boolean signals to numeric for correlation
                        signal_numeric = df[signal_col].astype(int)

                        # Calculate correlation where both have data
                        common_idx = indicator_data.index.intersection(signal_numeric.index)
                        if len(common_idx) > 10:  # Need sufficient data points
                            corr = indicator_data.loc[common_idx].corr(signal_numeric.loc[common_idx])
                            if not pd.isna(corr):
                                correlations[signal_col] = round(corr, 4)

                if correlations:
                    correlation_analysis[indicator] = correlations

        analysis_results["indicator_signal_correlations"] = correlation_analysis

        # Clear the timeout
        signal.alarm(0)
        return json.dumps(analysis_results, indent=2, default=str)

    except TimeoutError as e:
        signal.alarm(0)  # Clear the timeout
        logger.error(f"Trade decision analysis timed out: {e}")
        return json.dumps({"error": f"Analysis timed out: {str(e)}"})
    except Exception as e:
        signal.alarm(0)  # Clear the timeout
        logger.error(f"Error analyzing trade decision factors: {e}")
        return json.dumps({"error": f"Failed to analyze trade decisions: {str(e)}"})

@mcp.tool()
def get_current_indicator_snapshot(pair: str, timeframe: str = "5m", include_history: int = 10) -> str:
    """
    Get a snapshot of current indicator values and their recent history for a trading pair.
    This helps understand the current market state according to the strategy's indicators.

    Args:
        pair: Trading pair (e.g., 'BTC/USDT:USDT')
        timeframe: Timeframe for analysis (default: '5m')
        include_history: Number of recent periods to include for trend analysis
    """
    try:
        config = get_config()
        strategy = get_strategy()

        if strategy is None:
            return json.dumps({"error": "Strategy not loaded"})

        # Load recent pair data
        data_location = config.get("datadir", Path("user_data/data"))
        candles = load_pair_history(
            datadir=data_location,
            timeframe=timeframe,
            pair=pair,
            data_format="feather",
            candle_type=CandleType.FUTURES,
        )

        if candles.empty:
            return json.dumps({"error": f"No data found for {pair}"})

        # Get recent data including history
        recent_candles = candles.tail(include_history + 50)  # Extra buffer for indicator calculation

        # Generate indicators
        try:
            df = analyze_with_strategy(strategy, recent_candles, pair)
        except Exception as e:
            return json.dumps({"error": f"Strategy analysis failed: {str(e)}"})

        if df.empty:
            return json.dumps({"error": "No indicator data generated"})

        # Get the most recent data
        latest_data = df.tail(include_history)
        current_row = df.iloc[-1]

        # Identify columns
        base_columns = ["open", "high", "low", "close", "volume", "date"]
        signal_columns = ["enter_long", "enter_short", "exit_long", "exit_short"]
        indicator_columns = [col for col in df.columns
                           if col not in base_columns + signal_columns]

        # Current market state
        current_state = {
            "timestamp": str(current_row.name),
            "price": {
                "open": float(current_row["open"]),
                "high": float(current_row["high"]),
                "low": float(current_row["low"]),
                "close": float(current_row["close"]),
                "volume": float(current_row["volume"])
            },
            "signals": {
                "enter_long": bool(current_row.get("enter_long", False)),
                "enter_short": bool(current_row.get("enter_short", False)),
                "exit_long": bool(current_row.get("exit_long", False)),
                "exit_short": bool(current_row.get("exit_short", False))
            },
            "indicators": {}
        }

        # Analyze each indicator
        indicator_analysis = {}
        for indicator in indicator_columns:
            if indicator in latest_data.columns:
                indicator_series = latest_data[indicator].dropna()

                if not indicator_series.empty:
                    current_value = current_row[indicator] if pd.notna(current_row[indicator]) else None

                    analysis = {
                        "current_value": current_value,
                        "data_type": str(indicator_series.dtype)
                    }

                    if pd.api.types.is_numeric_dtype(indicator_series) and current_value is not None:
                        # Numeric indicator analysis
                        history_values = indicator_series.values

                        # Calculate trend strength safely
                        trend_strength = 0
                        if len(history_values) >= 2:
                            try:
                                val1 = float(history_values[-1])
                                val2 = float(history_values[-2])
                                trend_strength = abs(val1 - val2)
                            except (ValueError, TypeError):
                                trend_strength = 0

                        # Calculate percentile rank safely
                        percentile_rank = 0
                        try:
                            if pd.api.types.is_numeric_dtype(indicator_series):
                                percentile_rank = float((indicator_series <= current_value).sum() / len(indicator_series) * 100)
                        except (ValueError, TypeError):
                            percentile_rank = 0

                        analysis.update({
                            "current_value": float(current_value),
                            "recent_history": [float(x) for x in history_values[-min(5, len(history_values)):] if pd.notna(x)],
                            "trend": {
                                "direction": "up" if len(history_values) >= 2 and history_values[-1] > history_values[-2] else "down" if len(history_values) >= 2 else "stable",
                                "strength": trend_strength,
                                "consistency": "consistent" if len(set([
                                    "up" if history_values[i] > history_values[i-1] else "down"
                                    for i in range(1, min(len(history_values), 4))
                                ])) <= 1 else "mixed"
                            },
                            "statistics": {
                                "mean": float(indicator_series.mean()),
                                "std": float(indicator_series.std()),
                                "min": float(indicator_series.min()),
                                "max": float(indicator_series.max()),
                                "percentile_rank": percentile_rank
                            }
                        })

                        # Determine if indicator is in extreme territory
                        percentile = analysis["statistics"]["percentile_rank"]
                        if percentile >= 90:
                            analysis["territory"] = "extremely_high"
                        elif percentile >= 75:
                            analysis["territory"] = "high"
                        elif percentile <= 10:
                            analysis["territory"] = "extremely_low"
                        elif percentile <= 25:
                            analysis["territory"] = "low"
                        else:
                            analysis["territory"] = "normal"

                    else:
                        # Non-numeric indicator (boolean, categorical)
                        # Convert value_counts to dict with string keys to avoid JSON serialization issues
                        value_counts = indicator_series.value_counts()
                        value_counts_dict = {str(k): int(v) for k, v in value_counts.items()}

                        analysis.update({
                            "current_value": str(current_value) if current_value is not None else None,
                            "recent_history": [str(x) for x in indicator_series.tail(5).values if pd.notna(x)],
                            "value_counts": value_counts_dict
                        })

                    indicator_analysis[indicator] = analysis
                    current_state["indicators"][indicator] = current_value

        # Signal context analysis
        signal_context = {}
        active_signals = [signal for signal, active in current_state["signals"].items() if active]

        if active_signals:
            signal_context["active_signals"] = active_signals
            signal_context["indicator_states_during_signals"] = {}

            for signal in active_signals:
                signal_context["indicator_states_during_signals"][signal] = {
                    indicator: {
                        "value": current_state["indicators"].get(indicator),
                        "territory": indicator_analysis.get(indicator, {}).get("territory", "unknown"),
                        "trend": indicator_analysis.get(indicator, {}).get("trend", {}).get("direction", "unknown")
                    }
                    for indicator in indicator_columns[:10]  # Limit to first 10 indicators for readability
                }

        result = {
            "pair": pair,
            "timeframe": timeframe,
            "strategy": config.get("strategy", "Unknown"),
            "snapshot_time": datetime.now().isoformat(),
            "current_state": current_state,
            "indicator_analysis": indicator_analysis,
            "signal_context": signal_context,
            "summary": {
                "total_indicators": len(indicator_columns),
                "active_signals": len(active_signals),
                "indicators_in_extreme_territory": len([
                    ind for ind, analysis in indicator_analysis.items()
                    if analysis.get("territory") in ["extremely_high", "extremely_low"]
                ]),
                "trending_indicators": len([
                    ind for ind, analysis in indicator_analysis.items()
                    if analysis.get("trend", {}).get("consistency") == "consistent"
                ])
            }
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error getting indicator snapshot: {e}")
        return json.dumps({"error": f"Failed to get indicator snapshot: {str(e)}"})

@mcp.tool()
def calculate_performance_metrics(pair: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    Calculate detailed performance metrics for trades.

    Args:
        pair: Optional trading pair to filter by
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)
    """
    try:
        _, trades = get_backtest_data()

        if trades is None or trades.empty:
            return json.dumps({"error": "No trade data available"})

        # Apply filters
        filtered_trades = trades.copy()

        if pair:
            filtered_trades = filtered_trades[filtered_trades["pair"] == pair]

        if start_date:
            filtered_trades = filtered_trades[filtered_trades["open_date"] >= start_date]

        if end_date:
            filtered_trades = filtered_trades[filtered_trades["close_date"] <= end_date]

        if filtered_trades.empty:
            return json.dumps({"error": "No trades found with the specified filters"})

        # Calculate metrics
        total_trades = len(filtered_trades)
        winning_trades = len(filtered_trades[filtered_trades["profit_ratio"] > 0])
        losing_trades = len(filtered_trades[filtered_trades["profit_ratio"] <= 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_profit = filtered_trades["profit_abs"].sum()
        avg_profit = filtered_trades["profit_abs"].mean()

        winning_profit = filtered_trades[filtered_trades["profit_ratio"] > 0]["profit_abs"].sum()
        losing_profit = filtered_trades[filtered_trades["profit_ratio"] <= 0]["profit_abs"].sum()

        avg_winning_profit = filtered_trades[filtered_trades["profit_ratio"] > 0]["profit_abs"].mean() if winning_trades > 0 else 0
        avg_losing_profit = filtered_trades[filtered_trades["profit_ratio"] <= 0]["profit_abs"].mean() if losing_trades > 0 else 0

        profit_factor = abs(winning_profit / losing_profit) if losing_profit != 0 else float('inf')

        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for _, trade in filtered_trades.iterrows():
            if trade["profit_ratio"] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        metrics = {
            "filter_criteria": {
                "pair": pair or "All pairs",
                "start_date": start_date or "No start filter",
                "end_date": end_date or "No end filter"
            },
            "basic_metrics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_percent": round(win_rate, 2)
            },
            "profit_metrics": {
                "total_profit": round(total_profit, 4),
                "average_profit_per_trade": round(avg_profit, 4),
                "total_winning_profit": round(winning_profit, 4),
                "total_losing_profit": round(losing_profit, 4),
                "average_winning_profit": round(avg_winning_profit, 4),
                "average_losing_profit": round(avg_losing_profit, 4),
                "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "Infinite"
            },
            "streak_metrics": {
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses
            },
            "trade_duration": {
                "average_duration": str(filtered_trades["trade_duration"].mean()),
                "min_duration": str(filtered_trades["trade_duration"].min()),
                "max_duration": str(filtered_trades["trade_duration"].max())
            }
        }

        return json.dumps(metrics, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return json.dumps({"error": f"Failed to calculate metrics: {str(e)}"})

@mcp.tool()
def chat_with_trading_data(query: str, pair: Optional[str] = None) -> str:
    """
    Chat with your trading data using natural language queries.
    This tool provides a simplified pandasai-like interface for data analysis.

    Args:
        query: Natural language question about your trading data
        pair: Optional trading pair to filter the analysis
    """
    try:
        _, trades = get_backtest_data()

        if trades is None or trades.empty:
            return json.dumps({"error": "No trade data available for analysis"})

        # Filter by pair if specified
        if pair:
            trades = trades[trades["pair"] == pair]
            if trades.empty:
                return json.dumps({"error": f"No trades found for pair {pair}"})

        # Import the data analysis tools
        from user_data.data_analysis_tools import FreqtradeDataAnalyzer

        # Create analyzer
        analyzer = FreqtradeDataAnalyzer(trades)

        # Process the query
        result = analyzer.chat_with_data(query)

        return result

    except ImportError:
        return json.dumps({
            "error": "Data analysis tools not available. Please ensure data_analysis_tools.py is in the user_data directory."
        })
    except Exception as e:
        logger.error(f"Error in chat analysis: {e}")
        return json.dumps({"error": f"Analysis failed: {str(e)}"})

@mcp.tool()
def generate_trading_report(pair: Optional[str] = None, format: str = "summary") -> str:
    """
    Generate a comprehensive trading report.

    Args:
        pair: Optional trading pair to focus the report on
        format: Report format - "summary", "detailed", or "json"
    """
    try:
        _, trades = get_backtest_data()

        if trades is None or trades.empty:
            return json.dumps({"error": "No trade data available for report"})

        # Filter by pair if specified
        if pair:
            trades = trades[trades["pair"] == pair]
            if trades.empty:
                return json.dumps({"error": f"No trades found for pair {pair}"})

        # Calculate comprehensive metrics
        total_trades = len(trades)
        winning_trades = len(trades[trades["profit_ratio"] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_profit = trades["profit_abs"].sum()
        avg_profit = trades["profit_abs"].mean()

        # Risk metrics
        max_profit = trades["profit_abs"].max()
        max_loss = trades["profit_abs"].min()

        # Time analysis
        trades['trade_duration_minutes'] = pd.to_timedelta(trades['trade_duration']).dt.total_seconds() / 60
        avg_duration = trades['trade_duration_minutes'].mean()

        # Pair analysis
        if not pair:  # Only if not filtered by pair
            pair_performance = trades.groupby('pair').agg({
                'profit_abs': ['count', 'sum', 'mean'],
                'profit_ratio': lambda x: (x > 0).sum() / len(x) * 100
            }).round(4)
            pair_performance.columns = ['trades', 'total_profit', 'avg_profit', 'win_rate']
            pair_performance = pair_performance.sort_values('total_profit', ascending=False)

        # Exit reason analysis
        exit_analysis = trades.groupby('exit_reason').agg({
            'profit_abs': ['count', 'sum', 'mean']
        }).round(4)
        exit_analysis.columns = ['count', 'total_profit', 'avg_profit']

        report = {
            "report_title": f"Trading Performance Report{' - ' + pair if pair else ''}",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": str(trades["open_date"].min()),
                "end": str(trades["close_date"].max()),
                "days": (trades["close_date"].max() - trades["open_date"].min()).days
            },
            "overview": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_percent": round(win_rate, 2),
                "total_profit": round(total_profit, 4),
                "average_profit_per_trade": round(avg_profit, 4),
                "best_trade": round(max_profit, 4),
                "worst_trade": round(max_loss, 4),
                "average_trade_duration_minutes": round(avg_duration, 1)
            }
        }

        if format == "detailed":
            report["detailed_analysis"] = {
                "exit_reasons": exit_analysis.to_dict(orient='index'),
                "monthly_performance": trades.groupby(trades['open_date'].dt.to_period('M'))['profit_abs'].sum().to_dict(),
                "best_trades": trades.nlargest(5, 'profit_abs')[['pair', 'profit_abs', 'open_date', 'exit_reason']].to_dict(orient='records'),
                "worst_trades": trades.nsmallest(5, 'profit_abs')[['pair', 'profit_abs', 'open_date', 'exit_reason']].to_dict(orient='records')
            }

            if not pair:
                report["detailed_analysis"]["pair_performance"] = pair_performance.to_dict(orient='index')

        if format == "summary":
            # Create a readable summary
            summary_text = f"""
Trading Performance Report{' - ' + pair if pair else ''}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}% ({winning_trades} wins, {losing_trades} losses)
• Total Profit: {total_profit:.4f} USDT
• Average Profit: {avg_profit:.4f} USDT per trade
• Best Trade: +{max_profit:.4f} USDT
• Worst Trade: {max_loss:.4f} USDT
• Average Duration: {avg_duration:.1f} minutes

PERIOD: {trades['open_date'].min().strftime('%Y-%m-%d')} to {trades['close_date'].max().strftime('%Y-%m-%d')}
            """.strip()

            return summary_text

        return json.dumps(report, indent=2, default=str)

    except Exception as e:
        logger.error(f"Error generating trading report: {e}")
        return json.dumps({"error": f"Failed to generate report: {str(e)}"})

if __name__ == "__main__":
    mcp.run()
