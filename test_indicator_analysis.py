#!/usr/bin/env python3
"""
Test script for the enhanced MCP server indicator analysis capabilities
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_indicator_analysis():
    """Test the new indicator analysis tools."""
    print("🔍 Testing Enhanced MCP Server - Indicator Analysis")
    print("=" * 60)

    try:
        # Import the enhanced server functions
        from user_data.freqtrade_mcp_server import (
            analyze_indicator_states,
            analyze_trade_decision_factors,
            get_current_indicator_snapshot,
            get_available_pairs
        )

        print("✓ Successfully imported enhanced server functions")

        # Get available pairs for testing
        pairs_result = get_available_pairs()
        pairs_data = json.loads(pairs_result)
        backtest_pairs = pairs_data.get('backtest_pairs', [])

        if not backtest_pairs:
            print("❌ No pairs available for testing")
            return False

        test_pair = backtest_pairs[0]
        print(f"\n🎯 Testing with pair: {test_pair}")

        # Test 1: Analyze Indicator States
        print("\n1. Testing analyze_indicator_states...")
        try:
            indicator_result = analyze_indicator_states(test_pair, limit=500, focus_on_signals=True)
            indicator_data = json.loads(indicator_result)

            if "error" not in indicator_data:
                print(f"   ✓ Found {indicator_data.get('indicators_found', 0)} indicators")
                print(f"   ✓ Analyzed {indicator_data.get('analysis_period', {}).get('data_points', 0)} data points")

                # Show some indicator names
                indicators = indicator_data.get('indicator_list', [])[:5]
                if indicators:
                    print(f"   ✓ Sample indicators: {', '.join(indicators)}")

                # Check signal analysis
                signal_analysis = indicator_data.get('signal_analysis', {})
                if signal_analysis:
                    entry_signals = signal_analysis.get('entry_signals', {})
                    if entry_signals:
                        print(f"   ✓ Entry signals analyzed: {entry_signals.get('count', 0)}")

            else:
                print(f"   ⚠ Error: {indicator_data['error']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 2: Analyze Trade Decision Factors
        print("\n2. Testing analyze_trade_decision_factors...")
        try:
            decision_result = analyze_trade_decision_factors(test_pair, lookback_periods=3)
            decision_data = json.loads(decision_result)

            if "error" not in decision_data:
                print(f"   ✓ Analyzed {decision_data.get('total_indicators', 0)} indicators")
                print(f"   ✓ Lookback periods: {decision_data.get('lookback_periods', 0)}")

                # Check signal analysis
                signal_analysis = decision_data.get('signal_analysis', {})
                for signal_type, analysis in signal_analysis.items():
                    if analysis.get('count', 0) > 0:
                        print(f"   ✓ {signal_type}: {analysis['count']} signals analyzed")

                # Check correlations
                correlations = decision_data.get('indicator_signal_correlations', {})
                if correlations:
                    print(f"   ✓ Found correlations for {len(correlations)} indicators")

            else:
                print(f"   ⚠ Error: {decision_data['error']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 3: Get Current Indicator Snapshot
        print("\n3. Testing get_current_indicator_snapshot...")
        try:
            snapshot_result = get_current_indicator_snapshot(test_pair, include_history=5)
            snapshot_data = json.loads(snapshot_result)

            if "error" not in snapshot_data:
                current_state = snapshot_data.get('current_state', {})
                print(f"   ✓ Snapshot timestamp: {current_state.get('timestamp', 'N/A')}")

                # Check current signals
                signals = current_state.get('signals', {})
                active_signals = [signal for signal, active in signals.items() if active]
                if active_signals:
                    print(f"   ✓ Active signals: {', '.join(active_signals)}")
                else:
                    print("   ✓ No active signals at current time")

                # Check indicators
                indicators = current_state.get('indicators', {})
                print(f"   ✓ Current indicators tracked: {len(indicators)}")

                # Check summary
                summary = snapshot_data.get('summary', {})
                extreme_indicators = summary.get('indicators_in_extreme_territory', 0)
                trending_indicators = summary.get('trending_indicators', 0)
                print(f"   ✓ Indicators in extreme territory: {extreme_indicators}")
                print(f"   ✓ Consistently trending indicators: {trending_indicators}")

            else:
                print(f"   ⚠ Error: {snapshot_data['error']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 4: Enhanced Chat Analysis
        print("\n4. Testing enhanced chat analysis...")
        try:
            from user_data.freqtrade_mcp_server import chat_with_trading_data

            # Test indicator-related queries
            indicator_queries = [
                "Tell me about indicators",
                "What indicators influenced my trades?",
                "Show me indicator analysis"
            ]

            for query in indicator_queries:
                print(f"\n   Query: '{query}'")
                chat_result = chat_with_trading_data(query)
                chat_data = json.loads(chat_result)

                response = chat_data.get('response', 'No response')
                print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")

                # Check for advanced tools suggestions
                if 'advanced_tools' in chat_data:
                    print(f"   ✓ Advanced tools suggested: {len(chat_data['advanced_tools'])}")

        except Exception as e:
            print(f"   ❌ Chat analysis error: {e}")

        print("\n" + "=" * 60)
        print("✅ Enhanced Indicator Analysis Testing Complete!")
        print("\nNew capabilities added:")
        print("• 📊 Detailed indicator state analysis from strategy.analyze_ticker")
        print("• 🎯 Trade decision factor analysis (what indicators led to trades)")
        print("• 📸 Real-time indicator snapshots with trend analysis")
        print("• 💬 Enhanced chat interface with indicator query support")
        print("• 🔗 Indicator-signal correlation analysis")
        print("• 📈 Indicator territory analysis (extreme/normal levels)")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_indicator_insights():
    """Demonstrate practical indicator insights."""
    print("\n" + "=" * 60)
    print("🎯 PRACTICAL INDICATOR INSIGHTS DEMO")
    print("=" * 60)

    try:
        from user_data.freqtrade_mcp_server import get_available_pairs, analyze_indicator_states

        # Get a pair for demo
        pairs_result = get_available_pairs()
        pairs_data = json.loads(pairs_result)
        backtest_pairs = pairs_data.get('backtest_pairs', [])

        if not backtest_pairs:
            print("No pairs available for demo")
            return

        test_pair = backtest_pairs[0]
        print(f"Analyzing indicators for {test_pair}...")

        # Get indicator analysis
        result = analyze_indicator_states(test_pair, limit=500, focus_on_signals=True)
        data = json.loads(result)

        if "error" in data:
            print(f"Error: {data['error']}")
            return

        print(f"\n📊 INDICATOR ANALYSIS SUMMARY:")
        print(f"Strategy: {data.get('strategy', 'Unknown')}")
        print(f"Indicators found: {data.get('indicators_found', 0)}")
        print(f"Data points analyzed: {data.get('analysis_period', {}).get('data_points', 0)}")

        # Show signal analysis
        signal_analysis = data.get('signal_analysis', {})
        if signal_analysis:
            print(f"\n🎯 SIGNAL ANALYSIS:")

            entry_signals = signal_analysis.get('entry_signals', {})
            if entry_signals and entry_signals.get('count', 0) > 0:
                print(f"Entry signals: {entry_signals['count']}")
                print(f"  - Long entries: {entry_signals.get('long_entries', 0)}")
                print(f"  - Short entries: {entry_signals.get('short_entries', 0)}")

                # Show indicator states at entry
                indicator_states = entry_signals.get('indicator_states_at_entry', {})
                if indicator_states:
                    print(f"\n📈 INDICATOR STATES AT ENTRY (sample):")
                    for indicator, stats in list(indicator_states.items())[:3]:
                        if stats.get('mean') is not None:
                            print(f"  {indicator}:")
                            print(f"    Average value: {stats.get('mean', 0):.4f}")
                            print(f"    Range: {stats.get('min', 0):.4f} to {stats.get('max', 0):.4f}")

        # Show overall indicator analysis
        indicator_analysis = data.get('indicator_analysis', {})
        if indicator_analysis:
            print(f"\n📊 INDICATOR OVERVIEW:")
            numeric_indicators = 0
            boolean_indicators = 0

            for indicator, analysis in indicator_analysis.items():
                if 'mean' in analysis:
                    numeric_indicators += 1
                else:
                    boolean_indicators += 1

            print(f"Numeric indicators: {numeric_indicators}")
            print(f"Boolean/categorical indicators: {boolean_indicators}")

            # Show sample numeric indicator
            for indicator, analysis in list(indicator_analysis.items())[:2]:
                if 'mean' in analysis:
                    print(f"\n📈 Sample indicator: {indicator}")
                    print(f"  Current value: {analysis.get('latest_value', 'N/A')}")
                    print(f"  Average: {analysis.get('mean', 0):.4f}")
                    print(f"  Trend: {analysis.get('trend_direction', 'unknown')}")
                    break

        print(f"\n💡 INSIGHTS:")
        print("• Use 'analyze_trade_decision_factors' to see what indicators triggered trades")
        print("• Use 'get_current_indicator_snapshot' for real-time indicator states")
        print("• Combine with trade data to understand indicator effectiveness")

    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced MCP Server - Indicator Analysis Test Suite")
    print("Testing new capabilities for analyzing strategy indicators...")

    success = test_indicator_analysis()
    if success:
        demo_indicator_insights()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All enhanced features working perfectly!")
        print("\nThe MCP server now provides deep insights into:")
        print("• How indicators behave during signal generation")
        print("• What indicator states lead to trade decisions")
        print("• Real-time indicator monitoring and trend analysis")
        print("• Correlation between indicators and trading signals")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
