# Freqtrade MCP Server

A Model Context Protocol (MCP) server that provides access to freqtrade pair data and backtesting results. This server exposes freqtrade data through MCP resources and tools for analysis.

## Features

### Resources (Data Access)
- **freqtrade://config** - Get current freqtrade configuration
- **freqtrade://backtest/summary** - Get summary of latest backtest results
- **freqtrade://backtest/trades** - Get all trades from latest backtest
- **freqtrade://pairs/list** - Get list of available trading pairs
- **freqtrade://pair/{pair}/stats** - Get statistics for a specific trading pair

### Tools (Analysis Functions)
- **get_pair_data** - Load historical OHLCV data for a trading pair
- **analyze_trades_by_exit_reason** - Analyze trades grouped by exit reason
- **get_strategy_signals** - Generate trading signals using loaded strategy
- **calculate_performance_metrics** - Calculate detailed performance metrics

## Installation

1. Install the MCP Python SDK:
```bash
pip install "mcp[cli]"
```

2. Ensure you have freqtrade installed and configured with:
   - Configuration file at `user_data/config.json`
   - Historical data in your data directory
   - Backtest results in `user_data/backtest_results/`

## Usage

### Running the Server

#### Development Mode (with MCP Inspector)
```bash
cd /freqtrade
mcp dev user_data/freqtrade_mcp_server.py
```

#### Direct Execution
```bash
cd /freqtrade
python user_data/freqtrade_mcp_server.py
```

#### Install in Claude Desktop
```bash
cd /freqtrade
mcp install user_data/freqtrade_mcp_server.py --name "Freqtrade Data Server"
```

### Example Queries

#### Get Backtest Summary
Access the resource: `freqtrade://backtest/summary`

#### Get Pair Statistics
Access the resource: `freqtrade://pair/BTC/USDT:USDT/stats`

#### Load Pair Data
Use the tool: `get_pair_data` with parameters:
- pair: "BTC/USDT:USDT"
- timeframe: "5m"
- limit: 1000

#### Analyze Exit Reasons
Use the tool: `analyze_trades_by_exit_reason` with optional parameter:
- pair: "ETH/USDT:USDT" (optional)

#### Generate Strategy Signals
Use the tool: `get_strategy_signals` with parameters:
- pair: "BTC/USDT:USDT"
- timeframe: "5m"
- limit: 100

#### Calculate Performance Metrics
Use the tool: `calculate_performance_metrics` with optional parameters:
- pair: "BTC/USDT:USDT" (optional)
- start_date: "2025-01-01" (optional)
- end_date: "2025-01-31" (optional)

## Data Structures

### TradeInfo
- pair: Trading pair
- open_date/close_date: Trade timestamps
- profit_ratio/profit_abs: Profit metrics
- exit_reason: Why the trade was closed
- trade_duration: How long the trade lasted
- is_short: Whether it was a short trade
- open_rate/close_rate: Entry and exit prices
- amount/stake_amount: Trade size

### PairStats
- total_trades: Number of trades for the pair
- winning_trades/losing_trades: Win/loss counts
- win_rate: Percentage of winning trades
- total_profit/avg_profit: Profit metrics
- max_profit/min_profit: Best and worst trades

### BacktestSummary
- strategy_name: Name of the strategy used
- total_trades: Total number of trades
- win_rate: Overall win rate
- total_profit: Total profit/loss
- max_drawdown: Maximum drawdown
- start_date/end_date: Backtest period
- pairs: List of traded pairs

## Configuration

The server automatically loads configuration from:
1. `user_data/config.json` (primary)
2. Fallback defaults if config not found

Required directories:
- `user_data/data/` - Historical price data
- `user_data/backtest_results/` - Backtest results

## Error Handling

The server includes comprehensive error handling:
- Missing configuration files
- Missing data directories
- Invalid pair names
- Strategy loading failures
- Data loading errors

All errors are returned as JSON with descriptive error messages.

## Logging

The server logs important events and errors to help with debugging:
- Configuration loading
- Data loading status
- Strategy initialization
- Error conditions

## Extending the Server

To add new resources or tools:

1. Add new resource functions with `@mcp.resource("uri://path")` decorator
2. Add new tool functions with `@mcp.tool()` decorator
3. Use Pydantic models for structured responses
4. Include proper error handling and logging

## Troubleshooting

### Common Issues

1. **"No backtest data available"**
   - Ensure you have run a backtest and results are in `user_data/backtest_results/`

2. **"Strategy not loaded"**
   - Check that your strategy file is in the strategies directory
   - Verify the strategy name in your config.json

3. **"No data found for pair"**
   - Ensure historical data is downloaded for the pair
   - Check the pair format (e.g., "BTC/USDT:USDT" for futures)

4. **Configuration errors**
   - Verify `user_data/config.json` exists and is valid JSON
   - Check file permissions

### Debug Mode

Run with debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This MCP server is part of the freqtrade project and follows the same licensing terms.
