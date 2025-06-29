#!/usr/bin/env python3
"""
Demo script for the Freqtrade MCP Server

This script demonstrates all the capabilities of the MCP server including:
- Resources (data access)
- Tools (analysis functions)
- Chat-based data analysis
- Report generation
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def demo_resources():
    """Demonstrate MCP resources."""
    print("=" * 60)
    print("DEMO: MCP Resources (Data Access)")
    print("=" * 60)

    from user_data.freqtrade_mcp_server import (
        get_freqtrade_config,
        get_backtest_summary,
        get_available_pairs,
        get_all_trades
    )

    print("\n1. Freqtrade Configuration:")
    config = json.loads(get_freqtrade_config())
    print(f"   - Data directory: {config.get('datadir', 'N/A')}")
    exchange = config.get('exchange', {})
    if isinstance(exchange, dict):
        print(f"   - Exchange: {exchange.get('name', 'N/A')}")
    else:
        print(f"   - Exchange: {exchange}")
    print(f"   - Dry run: {config.get('dry_run', 'N/A')}")

    print("\n2. Backtest Summary:")
    summary = json.loads(get_backtest_summary())
    if "error" not in summary:
        print(f"   - Strategy: {summary.get('strategy_name', 'N/A')}")
        print(f"   - Total trades: {summary.get('total_trades', 0)}")
        print(f"   - Win rate: {summary.get('win_rate', 0):.1f}%")
        print(f"   - Total profit: {summary.get('total_profit', 0):.4f} USDT")
    else:
        print(f"   - Error: {summary['error']}")

    print("\n3. Available Pairs:")
    pairs = json.loads(get_available_pairs())
    backtest_pairs = pairs.get('backtest_pairs', [])
    print(f"   - Backtest pairs: {len(backtest_pairs)}")
    if backtest_pairs:
        print(f"   - Examples: {', '.join(backtest_pairs[:3])}")

    print("\n4. Trade Data Sample:")
    trades = json.loads(get_all_trades())
    if isinstance(trades, list) and trades:
        print(f"   - Total trades loaded: {len(trades)}")
        sample_trade = trades[0]
        print(f"   - Sample trade: {sample_trade['pair']} - {sample_trade['profit_abs']:.4f} USDT")
    else:
        print("   - No trade data available")

def demo_tools():
    """Demonstrate MCP tools."""
    print("\n" + "=" * 60)
    print("DEMO: MCP Tools (Analysis Functions)")
    print("=" * 60)

    from user_data.freqtrade_mcp_server import (
        get_pair_data,
        analyze_trades_by_exit_reason,
        calculate_performance_metrics,
        get_available_pairs
    )

    # Get a sample pair for testing
    pairs = json.loads(get_available_pairs())
    backtest_pairs = pairs.get('backtest_pairs', [])

    if not backtest_pairs:
        print("No pairs available for tool demonstration")
        return

    test_pair = backtest_pairs[0]
    print(f"\nUsing test pair: {test_pair}")

    print("\n1. Pair Data Loading:")
    pair_data = json.loads(get_pair_data(test_pair, limit=5))
    if "error" not in pair_data:
        print(f"   - Data points: {pair_data.get('data_points', 0)}")
        print(f"   - Date range: {pair_data.get('start_date', 'N/A')} to {pair_data.get('end_date', 'N/A')}")
    else:
        print(f"   - Error: {pair_data['error']}")

    print("\n2. Exit Reason Analysis:")
    exit_analysis = json.loads(analyze_trades_by_exit_reason())
    if "error" not in exit_analysis:
        exit_reasons = exit_analysis.get('exit_reasons', {})
        print(f"   - Exit reasons found: {len(exit_reasons)}")
        for reason, stats in list(exit_reasons.items())[:2]:
            print(f"   - {reason}: {stats.get('profit_ratio_count', 0)} trades")
    else:
        print(f"   - Error: {exit_analysis['error']}")

    print("\n3. Performance Metrics:")
    metrics = json.loads(calculate_performance_metrics())
    if "error" not in metrics:
        basic = metrics.get('basic_metrics', {})
        profit = metrics.get('profit_metrics', {})
        print(f"   - Total trades: {basic.get('total_trades', 0)}")
        print(f"   - Win rate: {basic.get('win_rate_percent', 0)}%")
        print(f"   - Profit factor: {profit.get('profit_factor', 'N/A')}")
    else:
        print(f"   - Error: {metrics['error']}")

def demo_chat_analysis():
    """Demonstrate chat-based data analysis."""
    print("\n" + "=" * 60)
    print("DEMO: Chat-based Data Analysis")
    print("=" * 60)

    from user_data.freqtrade_mcp_server import chat_with_trading_data

    # Sample queries to demonstrate
    queries = [
        "What was my total profit?",
        "What's my win rate?",
        "Which pair performed best?",
        "Show me my worst trades",
        "What are the most common exit reasons?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        try:
            result = json.loads(chat_with_trading_data(query))
            response = result.get('response', 'No response')
            print(f"   Answer: {response}")

            # Show additional details for some queries
            if 'details' in result:
                details = result['details']
                if isinstance(details, dict):
                    for key, value in list(details.items())[:2]:  # Show first 2 details
                        print(f"   - {key}: {value}")
        except Exception as e:
            print(f"   Error: {e}")

def demo_reports():
    """Demonstrate report generation."""
    print("\n" + "=" * 60)
    print("DEMO: Report Generation")
    print("=" * 60)

    from user_data.freqtrade_mcp_server import generate_trading_report

    print("\n1. Summary Report:")
    summary_report = generate_trading_report(format="summary")
    # Show first few lines of the report
    lines = summary_report.split('\n')[:10]
    for line in lines:
        print(f"   {line}")
    if len(summary_report.split('\n')) > 10:
        print("   ...")

    print("\n2. JSON Report (sample):")
    json_report = json.loads(generate_trading_report(format="json"))
    overview = json_report.get('overview', {})
    print(f"   - Report title: {json_report.get('report_title', 'N/A')}")
    print(f"   - Total trades: {overview.get('total_trades', 0)}")
    print(f"   - Win rate: {overview.get('win_rate_percent', 0)}%")
    print(f"   - Total profit: {overview.get('total_profit', 0):.4f} USDT")

def main():
    """Run the complete demo."""
    print("🚀 Freqtrade MCP Server Demonstration")
    print("This demo shows all capabilities of the MCP server")

    try:
        demo_resources()
        demo_tools()
        demo_chat_analysis()
        demo_reports()

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe MCP server provides:")
        print("• 📊 Data access through resources")
        print("• 🔧 Analysis tools for detailed metrics")
        print("• 💬 Natural language chat interface")
        print("• 📈 Comprehensive report generation")
        print("\nTo use the server:")
        print("1. Run: python user_data/freqtrade_mcp_server.py")
        print("2. Or with MCP inspector: mcp dev user_data/freqtrade_mcp_server.py")
        print("3. Or install in Claude Desktop: mcp install user_data/freqtrade_mcp_server.py")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
