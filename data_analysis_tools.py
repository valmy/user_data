#!/usr/bin/env python3
"""
Data Analysis Tools for Freqtrade MCP Server

This module provides pandas-based data analysis tools that can be used
with the MCP server to analyze trading data.
"""

import pandas as pd
import json
from typing import Dict

class FreqtradeDataAnalyzer:
    """A simple data analyzer for freqtrade data."""

    def __init__(self, trades_df: pd.DataFrame, pair_data: Dict[str, pd.DataFrame] = None):
        """
        Initialize the analyzer with trade data and optional pair data.

        Args:
            trades_df: DataFrame containing trade data
            pair_data: Dictionary of pair name -> OHLCV DataFrame
        """
        self.trades_df = trades_df
        self.pair_data = pair_data or {}

    def chat_with_data(self, query: str) -> str:
        """
        Process natural language queries about the trading data.
        This is a simplified version of what pandasai would do.

        Args:
            query: Natural language query about the data

        Returns:
            JSON string with analysis results
        """
        query_lower = query.lower()

        try:
            # Profit-related queries
            if any(word in query_lower for word in ['profit', 'loss', 'money', 'earn']):
                return self._analyze_profits(query_lower)

            # Win rate queries
            elif any(word in query_lower for word in ['win rate', 'success', 'winning']):
                return self._analyze_win_rates(query_lower)

            # Time-related queries
            elif any(word in query_lower for word in ['time', 'duration', 'long', 'short']):
                return self._analyze_time_patterns(query_lower)

            # Pair-related queries
            elif any(word in query_lower for word in ['pair', 'symbol', 'coin', 'btc', 'eth']):
                return self._analyze_pairs(query_lower)

            # Indicator-related queries
            elif any(word in query_lower for word in ['indicator', 'signal', 'rsi', 'macd', 'ema', 'sma', 'bollinger', 'stoch']):
                return self._analyze_indicators(query_lower)

            # Exit reason queries
            elif any(word in query_lower for word in ['exit', 'stop', 'reason', 'why']):
                return self._analyze_exit_reasons(query_lower)

            # General statistics
            elif any(word in query_lower for word in ['summary', 'overview', 'stats', 'statistics']):
                return self._general_summary()

            # Best/worst performance
            elif any(word in query_lower for word in ['best', 'worst', 'top', 'bottom']):
                return self._analyze_performance_extremes(query_lower)

            else:
                return json.dumps({
                    "response": "I can help you analyze your trading data. Try asking about profits, win rates, trading pairs, exit reasons, or general statistics.",
                    "suggestions": [
                        "What was my total profit?",
                        "Which pair performed best?",
                        "What's my win rate?",
                        "Show me exit reasons",
                        "What was my worst trade?"
                    ]
                })

        except Exception as e:
            return json.dumps({"error": f"Analysis failed: {str(e)}"})

    def _analyze_profits(self, query: str) -> str:
        """Analyze profit-related queries."""
        total_profit = self.trades_df['profit_abs'].sum()
        avg_profit = self.trades_df['profit_abs'].mean()
        winning_trades = self.trades_df[self.trades_df['profit_ratio'] > 0]
        losing_trades = self.trades_df[self.trades_df['profit_ratio'] <= 0]

        result = {
            "response": f"Your total profit is {total_profit:.4f} USDT with an average of {avg_profit:.4f} per trade.",
            "details": {
                "total_profit": round(total_profit, 4),
                "average_profit_per_trade": round(avg_profit, 4),
                "total_winning_profit": round(winning_trades['profit_abs'].sum(), 4),
                "total_losing_profit": round(losing_trades['profit_abs'].sum(), 4),
                "best_trade": round(self.trades_df['profit_abs'].max(), 4),
                "worst_trade": round(self.trades_df['profit_abs'].min(), 4)
            }
        }

        if 'best' in query or 'top' in query:
            best_trades = self.trades_df.nlargest(5, 'profit_abs')[['pair', 'profit_abs', 'open_date']]
            result["best_trades"] = best_trades.to_dict(orient='records')

        return json.dumps(result, indent=2, default=str)

    def _analyze_win_rates(self, query: str) -> str:
        """Analyze win rate queries."""
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['profit_ratio'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        result = {
            "response": f"Your overall win rate is {win_rate:.2f}% ({winning_trades} wins out of {total_trades} trades).",
            "details": {
                "win_rate_percent": round(win_rate, 2),
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "total_trades": total_trades
            }
        }

        # Add per-pair win rates if requested
        if 'pair' in query:
            pair_win_rates = []
            for pair in self.trades_df['pair'].unique():
                pair_trades = self.trades_df[self.trades_df['pair'] == pair]
                pair_wins = len(pair_trades[pair_trades['profit_ratio'] > 0])
                pair_total = len(pair_trades)
                pair_win_rate = (pair_wins / pair_total * 100) if pair_total > 0 else 0
                pair_win_rates.append({
                    "pair": pair,
                    "win_rate": round(pair_win_rate, 2),
                    "wins": pair_wins,
                    "total": pair_total
                })
            result["pair_win_rates"] = sorted(pair_win_rates, key=lambda x: x['win_rate'], reverse=True)

        return json.dumps(result, indent=2)

    def _analyze_time_patterns(self, query: str) -> str:
        """Analyze time-related patterns."""
        self.trades_df['trade_duration_minutes'] = pd.to_timedelta(self.trades_df['trade_duration']).dt.total_seconds() / 60

        avg_duration = self.trades_df['trade_duration_minutes'].mean()
        min_duration = self.trades_df['trade_duration_minutes'].min()
        max_duration = self.trades_df['trade_duration_minutes'].max()

        result = {
            "response": f"Average trade duration is {avg_duration:.1f} minutes (range: {min_duration:.1f} to {max_duration:.1f} minutes).",
            "details": {
                "average_duration_minutes": round(avg_duration, 1),
                "min_duration_minutes": round(min_duration, 1),
                "max_duration_minutes": round(max_duration, 1),
                "median_duration_minutes": round(self.trades_df['trade_duration_minutes'].median(), 1)
            }
        }

        # Analyze by time of day if requested
        if 'hour' in query or 'time of day' in query:
            self.trades_df['hour'] = pd.to_datetime(self.trades_df['open_date']).dt.hour
            hourly_stats = self.trades_df.groupby('hour').agg({
                'profit_abs': ['count', 'mean', 'sum']
            }).round(4)
            hourly_stats.columns = ['trades_count', 'avg_profit', 'total_profit']
            result["hourly_patterns"] = hourly_stats.to_dict(orient='index')

        return json.dumps(result, indent=2, default=str)

    def _analyze_pairs(self, query: str) -> str:
        """Analyze trading pair performance."""
        pair_stats = self.trades_df.groupby('pair').agg({
            'profit_abs': ['count', 'sum', 'mean'],
            'profit_ratio': lambda x: (x > 0).sum() / len(x) * 100
        }).round(4)

        pair_stats.columns = ['trades_count', 'total_profit', 'avg_profit', 'win_rate']
        pair_stats = pair_stats.sort_values('total_profit', ascending=False)

        best_pair = pair_stats.index[0]
        worst_pair = pair_stats.index[-1]

        result = {
            "response": f"Best performing pair: {best_pair} with {pair_stats.loc[best_pair, 'total_profit']:.4f} total profit. Worst: {worst_pair}.",
            "pair_performance": pair_stats.to_dict(orient='index')
        }

        return json.dumps(result, indent=2, default=str)

    def _analyze_indicators(self, query: str) -> str:
        """Analyze indicator-related queries."""
        # This is a simplified analysis since we don't have indicator data in trades_df
        # In a real implementation, this would connect to the MCP server's indicator tools

        result = {
            "response": "For detailed indicator analysis, please use the MCP server's indicator analysis tools.",
            "available_tools": [
                "analyze_indicator_states - Analyze indicator states from strategy.analyze_ticker",
                "analyze_trade_decision_factors - Understand what indicators led to trade decisions",
                "get_current_indicator_snapshot - Get current indicator values and trends"
            ],
            "suggestion": "Try asking: 'What indicators influenced my recent trades?' or 'Show me current indicator states for BTC/USDT'"
        }

        # Basic analysis from trade data
        if 'signal' in query or 'entry' in query:
            # Analyze entry patterns from available data
            entry_analysis = {
                "total_trades": len(self.trades_df),
                "trades_by_hour": self.trades_df.groupby(pd.to_datetime(self.trades_df['open_date']).dt.hour).size().to_dict(),
                "avg_trade_duration": str(pd.to_timedelta(self.trades_df['trade_duration']).mean()),
            }
            result["basic_entry_analysis"] = entry_analysis

        return json.dumps(result, indent=2, default=str)

    def _analyze_exit_reasons(self, query: str) -> str:
        """Analyze exit reasons."""
        exit_stats = self.trades_df.groupby('exit_reason').agg({
            'profit_abs': ['count', 'sum', 'mean'],
            'profit_ratio': lambda x: (x > 0).sum() / len(x) * 100
        }).round(4)

        exit_stats.columns = ['count', 'total_profit', 'avg_profit', 'win_rate']
        exit_stats = exit_stats.sort_values('count', ascending=False)

        most_common = exit_stats.index[0]

        result = {
            "response": f"Most common exit reason: {most_common} ({exit_stats.loc[most_common, 'count']} trades).",
            "exit_reason_analysis": exit_stats.to_dict(orient='index')
        }

        return json.dumps(result, indent=2, default=str)

    def _analyze_performance_extremes(self, query: str) -> str:
        """Analyze best and worst performing trades."""
        if 'best' in query or 'top' in query:
            best_trades = self.trades_df.nlargest(5, 'profit_abs')[
                ['pair', 'profit_abs', 'profit_ratio', 'open_date', 'exit_reason']
            ]
            result = {
                "response": f"Your best trade made {best_trades.iloc[0]['profit_abs']:.4f} profit on {best_trades.iloc[0]['pair']}.",
                "best_trades": best_trades.to_dict(orient='records')
            }
        else:  # worst
            worst_trades = self.trades_df.nsmallest(5, 'profit_abs')[
                ['pair', 'profit_abs', 'profit_ratio', 'open_date', 'exit_reason']
            ]
            result = {
                "response": f"Your worst trade lost {abs(worst_trades.iloc[0]['profit_abs']):.4f} on {worst_trades.iloc[0]['pair']}.",
                "worst_trades": worst_trades.to_dict(orient='records')
            }

        return json.dumps(result, indent=2, default=str)

    def _general_summary(self) -> str:
        """Provide a general summary of trading performance."""
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['profit_ratio'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = self.trades_df['profit_abs'].sum()

        result = {
            "response": f"Trading Summary: {total_trades} trades, {win_rate:.1f}% win rate, {total_profit:.4f} total profit.",
            "summary": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate_percent": round(win_rate, 2),
                "total_profit": round(total_profit, 4),
                "average_profit": round(self.trades_df['profit_abs'].mean(), 4),
                "best_trade": round(self.trades_df['profit_abs'].max(), 4),
                "worst_trade": round(self.trades_df['profit_abs'].min(), 4),
                "unique_pairs": self.trades_df['pair'].nunique(),
                "date_range": {
                    "start": str(self.trades_df['open_date'].min()),
                    "end": str(self.trades_df['close_date'].max())
                }
            }
        }

        return json.dumps(result, indent=2, default=str)

def create_analyzer_from_mcp_data(trades_json: str) -> FreqtradeDataAnalyzer:
    """
    Create a data analyzer from MCP server trade data.

    Args:
        trades_json: JSON string from MCP server containing trade data

    Returns:
        FreqtradeDataAnalyzer instance
    """
    trades_data = json.loads(trades_json)
    if isinstance(trades_data, list):
        trades_df = pd.DataFrame(trades_data)
    else:
        trades_df = pd.DataFrame()

    return FreqtradeDataAnalyzer(trades_df)
