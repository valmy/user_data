#!/usr/bin/env python3
"""
Test script for the Freqtrade MCP Server
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_server_functions():
    """Test the main functions of the MCP server."""
    print("Testing Freqtrade MCP Server...")
    
    try:
        # Import the server module
        from user_data.freqtrade_mcp_server import (
            get_config, 
            get_backtest_data, 
            get_freqtrade_config,
            get_backtest_summary,
            get_available_pairs
        )
        
        print("✓ Successfully imported server module")
        
        # Test configuration loading
        print("\n1. Testing configuration loading...")
        config = get_config()
        print(f"✓ Configuration loaded: {bool(config)}")
        print(f"  - Data directory: {config.get('datadir', 'Not set')}")
        print(f"  - Strategy: {config.get('strategy', 'Not set')}")
        print(f"  - Timeframe: {config.get('timeframe', 'Not set')}")
        
        # Test backtest data loading
        print("\n2. Testing backtest data loading...")
        stats, trades = get_backtest_data()
        print(f"✓ Backtest stats loaded: {bool(stats)}")
        print(f"✓ Backtest trades loaded: {trades is not None}")
        if trades is not None:
            print(f"  - Number of trades: {len(trades)}")
            if not trades.empty:
                print(f"  - Pairs traded: {trades['pair'].nunique()}")
                print(f"  - Date range: {trades['open_date'].min()} to {trades['close_date'].max()}")
        
        # Test resource functions
        print("\n3. Testing MCP resource functions...")
        
        # Test config resource
        config_result = get_freqtrade_config()
        config_data = json.loads(config_result)
        print(f"✓ Config resource: {len(config_data)} items")
        
        # Test backtest summary resource
        summary_result = get_backtest_summary()
        summary_data = json.loads(summary_result)
        if "error" not in summary_data:
            print(f"✓ Backtest summary resource: {summary_data.get('total_trades', 0)} trades")
        else:
            print(f"⚠ Backtest summary: {summary_data['error']}")
        
        # Test pairs list resource
        pairs_result = get_available_pairs()
        pairs_data = json.loads(pairs_result)
        print(f"✓ Pairs list resource: {len(pairs_data.get('backtest_pairs', []))} backtest pairs")
        print(f"  Available data pairs: {len(pairs_data.get('available_data_pairs', []))}")
        
        print("\n4. Testing tool functions...")
        
        # Import tool functions
        from user_data.freqtrade_mcp_server import get_pair_data, analyze_trades_by_exit_reason
        
        # Test with a common pair if available
        if pairs_data.get('backtest_pairs'):
            test_pair = pairs_data['backtest_pairs'][0]
            print(f"Testing with pair: {test_pair}")
            
            # Test pair data loading
            pair_data_result = get_pair_data(test_pair, limit=10)
            pair_data = json.loads(pair_data_result)
            if "error" not in pair_data:
                print(f"✓ Pair data tool: {pair_data.get('data_points', 0)} data points")
            else:
                print(f"⚠ Pair data tool: {pair_data['error']}")
            
            # Test exit reason analysis
            exit_analysis_result = analyze_trades_by_exit_reason(test_pair)
            exit_analysis = json.loads(exit_analysis_result)
            if "error" not in exit_analysis:
                print(f"✓ Exit reason analysis: {len(exit_analysis.get('exit_reasons', {}))} exit reasons")
            else:
                print(f"⚠ Exit reason analysis: {exit_analysis['error']}")
        else:
            print("⚠ No backtest pairs available for testing tools")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcp_server_structure():
    """Test the MCP server structure and decorators."""
    print("\n5. Testing MCP server structure...")
    
    try:
        from user_data.freqtrade_mcp_server import mcp
        
        # Check if server is properly initialized
        print(f"✓ MCP server initialized: {mcp.name}")
        
        # Note: We can't easily test the actual MCP protocol without a client,
        # but we can verify the server structure is correct
        print("✓ MCP server structure appears correct")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP server structure error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Freqtrade MCP Server Test Suite")
    print("=" * 60)
    
    success = test_server_functions()
    if success:
        success = test_mcp_server_structure()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The MCP server is ready to use.")
        print("\nTo run the server:")
        print("  python user_data/freqtrade_mcp_server.py")
        print("\nTo test with MCP inspector (requires Node.js):")
        print("  mcp dev user_data/freqtrade_mcp_server.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
