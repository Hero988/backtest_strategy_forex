import os
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from dotenv import load_dotenv
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pandas_ta as ta


# Load environment variables for MT5 login credentials
load_dotenv()
login = int(os.getenv('MT5_LOGIN'))  # Replace with your login ID
password = os.getenv('MT5_PASSWORD')  # Replace with your password
server = os.getenv('MT5_SERVER')  # Replace with your server name

# Initialize MetaTrader 5 connection
if not mt5.initialize(login=login, password=password, server=server):
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()

def fetch_historical_data(symbol, timeframe, n_bars):
    """
    Fetch historical data from MetaTrader 5.

    Parameters:
        symbol (str): The trading pair symbol (e.g., "EURUSD").
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1).
        n_bars (int): Number of bars to fetch.

    Returns:
        pd.DataFrame: Historical OHLC data.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print(f"Failed to fetch data for {symbol}")
        quit()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def generate_signals_with_trend(df):
    """
    Generate buy and sell signals based on SMA crossovers and trend direction.

    Parameters:
        df (DataFrame): Input DataFrame containing at least a 'close' column.
        short_window (int): Window size for the short SMA.
        long_window (int): Window size for the long SMA.

    Returns:
        DataFrame: The input DataFrame with additional columns for SMAs, signals, and positions.
    """
    short_window = 5
    long_window = 25
    df_copy = df.copy()

    # Calculate moving averages
    df_copy['SMA_short'] = df_copy['close'].rolling(window=short_window).mean()
    df_copy['SMA_long'] = df_copy['close'].rolling(window=long_window).mean()
    df_copy['SMA_trend'] = df_copy['close'].rolling(window=long_window * 2).mean()  # Longer SMA for trend direction

    # Generate signals based on crossovers and trend direction
    df_copy['signal'] = 0

    # Detect buy signal (short SMA crosses above long SMA and trend is bullish)
    df_copy.loc[
        (df_copy['SMA_short'] > df_copy['SMA_long']) &
        (df_copy['SMA_short'].shift(1) <= df_copy['SMA_long'].shift(1)) &
        (df_copy['close'] > df_copy['SMA_trend']),  # Bullish trend
        'signal'
    ] = 1

    # Detect sell signal (short SMA crosses below long SMA and trend is bearish)
    df_copy.loc[
        (df_copy['SMA_short'] < df_copy['SMA_long']) &
        (df_copy['SMA_short'].shift(1) >= df_copy['SMA_long'].shift(1)) &
        (df_copy['close'] < df_copy['SMA_trend']),  # Bearish trend
        'signal'
    ] = -1

    # Shift signal to apply on the next bar
    df_copy['position'] = df_copy['signal'].shift()

    return df_copy, short_window, long_window

def backtest_strategy(df, pair_folder, pair_output_dir, initial_balance=10000):
    """
    Backtest a simple SMA crossover strategy with stop loss and take profit.

    Parameters:
        df (DataFrame): Historical OHLC data.
        pair_folder (str): Folder to save trade logs and statistics.
        pair_output_dir (str): Directory to save realistic environment logs.
        initial_balance (float): Starting balance for the backtest.
        stop_loss (float): Stop loss in pips.
        take_profit (float): Take profit in pips.

    Returns:
        equity_history (list): Equity balance over time.
        drawdown_history (list): Drawdown history over time.
        df (DataFrame): Updated DataFrame with signals and positions.
    """
    stop_loss=500 
    take_profit=1000

    df_copy, short_window, long_window = generate_signals_with_trend(df)
    
    # Backtesting logic
    equity = initial_balance
    max_equity = equity
    point = mt5.symbol_info(pair_folder).point
    position = None
    equity_history = [equity]
    drawdown_history = [0]
    trades = []  # To log trades

    # Adjust lot size for JPY pairs
    lot_size = 10000  # Default lot size
    if "JPY" in pair_folder:
        lot_size = 1000  # Adjust to a smaller, more reasonable lot size for JPY pairs

    # Define the commission per full lot (100,000 units)
    commission_per_lot = 3  # $3 per lot

    # Calculate commission
    commission = commission_per_lot * (lot_size / 100000)

    for i in range(1, len(df_copy)):
        row = df_copy.iloc[i]

        # Check for an active position
        if position is not None:
            if position['type'] == 'sell':
                # Adjust for partial fills
                fill_ratio = 0.95  # Assume 95% of the order gets filled

                # Check take profit for sell
                if row['low'] <= position['take_profit']:
                    profit = ((position['entry'] - position['take_profit']) * lot_size) - commission
                    equity += profit 
                    trades.append({'Type': 'Sell', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': profit, 'Entry': position['entry'],
                                   'Exit': position['take_profit'], 'Result': 'Win', 'Equity': equity})
                    position = None  # Close the position
                # Check stop loss for sell
                elif row['high'] >= position['stop_loss']:
                    loss = ((position['stop_loss'] - position['entry']) * lot_size) - commission
                    equity -= loss 
                    trades.append({'Type': 'Sell', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': -loss, 'Entry': position['entry'],
                                   'Exit': position['stop_loss'], 'Result': 'Loss', 'Equity': equity})
                    position = None  # Close the position

            elif position['type'] == 'buy':
                # Adjust for partial fills
                fill_ratio = 0.95  # Assume 95% of the order gets filled

                # Check take profit for buy
                if row['high'] >= position['take_profit']:
                    profit = ((position['take_profit'] - position['entry']) * lot_size) - commission
                    equity += profit
                    trades.append({'Type': 'Buy', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': profit, 'Entry': position['entry'],
                                   'Exit': position['take_profit'], 'Result': 'Win', 'Equity': equity})
                    position = None  # Close the position
                # Check stop loss for buy
                elif row['low'] <= position['stop_loss']:
                    loss = ((position['entry'] - position['stop_loss']) * lot_size) - commission
                    equity -= loss
                    trades.append({'Type': 'Buy', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': -loss, 'Entry': position['entry'],
                                   'Exit': position['stop_loss'], 'Result': 'Loss', 'Equity': equity})
                    position = None  # Close the position

        # Open a new position
        if position is None:
            if row['position'] == 1:  # Buy position
                position = {
                    'type': 'buy',
                    'stop_loss': row['close'] - stop_loss * point,
                    'take_profit': row['close'] + take_profit * point,
                    'entry': row['close'],
                    'entry_time': row['time']
                }
            elif row['position'] == -1:  # Sell position
                position = {
                    'type': 'sell',
                    'stop_loss': row['close'] + stop_loss * point,
                    'take_profit': row['close'] - take_profit * point,
                    'entry': row['close'],
                    'entry_time': row['time']
                }

        # Track equity and drawdown
        equity_history.append(equity)
        max_equity = max(max_equity, equity)
        drawdown_history.append(((max_equity - equity) / max_equity) * 100)

        # Stop backtest if equity falls below a threshold
        if equity_history[-1] < initial_balance - 1000:
            break

    # Calculate statistics
    total_trades = len(trades)
    wins = sum(1 for trade in trades if trade['Result'] == 'Win')
    win_percentage = (wins / total_trades) * 100 if total_trades > 0 else 0

    stats = {
        'Total Trades': total_trades,
        'Wins': wins,
        'Losses': total_trades - wins,
        'Win Percentage': win_percentage,
        'Final Equity': equity
    }

    # Save trade log and statistics
    trade_log_df = pd.DataFrame(trades)
    if not os.path.exists(pair_output_dir):
        os.makedirs(pair_output_dir)
    trade_log_path = os.path.join(pair_output_dir, 'trade_log.csv')
    stats_path = os.path.join(pair_output_dir, 'statistics.txt')

    trade_log_df.to_csv(trade_log_path, index=False)
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value}\n')

    return equity_history, drawdown_history, df_copy, stats

def plot_results(equity_history, drawdown_history, df, symbol, output_dir):
    """
    Plot the backtesting results.

    Parameters:
        equity_history (list): Equity balance over time.
        drawdown_history (list): Drawdown history over time.
        df (DataFrame): Updated DataFrame with signals and positions.
        symbol (str): Trading symbol.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot equity
    axs[0].plot(equity_history, label="Equity", color='blue')
    axs[0].set_title(f"Equity Curve ({symbol})")
    axs[0].set_ylabel("Equity")
    axs[0].legend()
    axs[0].grid()

    # Plot drawdown
    axs[1].plot(drawdown_history, label="Drawdown (%)", color='red', linestyle='--')
    axs[1].set_title("Drawdown Over Time")
    axs[1].set_ylabel("Drawdown (%)")
    axs[1].legend()
    axs[1].grid()

    # Plot price with signals
    axs[2].plot(df['close'], label="Price", color='black')
    axs[2].plot(df['SMA_short'], label="SMA Short", color='green', linestyle='--')
    axs[2].plot(df['SMA_long'], label="SMA Long", color='red', linestyle='--')
    buy_signals = df[df['signal'] == 1].index
    sell_signals = df[df['signal'] == -1].index
    axs[2].scatter(buy_signals, df.loc[buy_signals]['close'], label="Buy Signal", marker="^", color="green", alpha=1)
    axs[2].scatter(sell_signals, df.loc[sell_signals]['close'], label="Sell Signal", marker="v", color="red", alpha=1)
    axs[2].set_title("Price and Signals")
    axs[2].set_ylabel("Price")
    axs[2].legend()
    axs[2].grid()

    plt.xlabel("Time")
    # Save the combined plot
    plot_file = os.path.join(output_dir, f"{symbol}_equity_drawdown_plot.png")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

def process_and_segment_data(full_data_path, days_for_backtest):
    """
    Reads data from a file, applies the RSI indicator, and segments the last year's data.
    
    Args:
        full_data_path (str): The path to the full dataset (CSV file).
    
    Returns:
        pd.DataFrame: A DataFrame containing the last year's data with the RSI indicator applied.
    """
    # Read data into a DataFrame
    df = pd.read_csv(full_data_path)
    
    # Ensure the 'time' column is in datetime format and set it as the index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        raise ValueError("The dataset must contain a 'time' column.")

    # Apply the RSI indicator to the DataFrame
    if 'close' in df.columns:
        df['RSI'] = ta.rsi(df['close'], length=14)
    else:
        raise ValueError("The dataset must contain a 'close' column.")

    # Segment data for the last year
    backtest_data_date = (datetime.now() - timedelta(days=days_for_backtest))
    # Format the date to "YYYY-MM-DD 00:00:00"
    backtest_data_date = backtest_data_date.replace(hour=0, minute=0, second=0, microsecond=0)

    backtest_data_segmented = df[df.index >= backtest_data_date]

    print(backtest_data_segmented.tail())

    time.sleep(6000)

    return backtest_data_segmented

def main_function(data_dir, output_dir, days_for_backtest):
    # Initialize a list to store results
    final_equity_results = []
    os.makedirs(output_dir, exist_ok=True)  # Ensure the main output directory exists

    for pair_folder in os.listdir(data_dir):
        pair_path = os.path.join(data_dir, pair_folder)
        if not os.path.isdir(pair_path):
            continue

        # Create a subfolder inside output_dir for the pair_folder
        pair_output_dir = os.path.join(output_dir, pair_folder)
        os.makedirs(pair_output_dir, exist_ok=True)

        # Load Data
        full_data_path = os.path.join(pair_path, f"{pair_folder}_all_data.csv")
        if not os.path.exists(full_data_path):
            print(f"Data files missing for {pair_folder}")
            continue

        df_backtest =  process_and_segment_data(full_data_path, days_for_backtest)

        equity_history, drawdown_history, df_copy, stats = backtest_strategy(df_backtest ,pair_folder, pair_output_dir, initial_balance=10000)
        plot_results(equity_history, drawdown_history, df_copy, pair_folder, pair_output_dir)

        # Assuming stats is available after running the function
        final_equity = stats['Final Equity']

        # Append results to the list
        final_equity_results.append({'Pair Folder': pair_folder, 'Final Equity': final_equity})

        print(f"Completed backtest {pair_folder}")

    # Create a DataFrame from the results
    final_equity_df = pd.DataFrame(final_equity_results)

    # Sort the DataFrame by 'Final Equity' in descending order
    final_equity_df = final_equity_df.sort_values(by='Final Equity', ascending=False)

    final_equity_results_path = os.path.join(output_dir, 'final_equity_summary.csv')

    # Save the DataFrame to a CSV or print
    final_equity_df.to_csv(final_equity_results_path, index=False)

def get_candles_from_start_date(symbol, timeframe="1h", start_date="2024-01-01"):
    """
    Retrieve historical candles for a given symbol and timeframe starting from a specific date.

    Parameters:
        symbol (str): The trading symbol (e.g., "EURUSD").
        timeframe (str): Timeframe for the candles ("1m", "5m", "15m", "1h", "4h", "1d").
        start_date (str): Start date in "YYYY-MM-DD" format.

    Returns:
        DataFrame: Historical candle data with time and prices.
    """
    timeframes = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    if timeframe not in timeframes:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Valid options: {list(timeframes.keys())}")

    # Parse the start date
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.now() + timedelta(hours=1)

    # Get rates from MetaTrader 5
    rates = mt5.copy_rates_range(symbol, timeframes[timeframe], start_date_obj, end_date_obj)

    if rates is None or len(rates) == 0:
        print(f"No data retrieved for {symbol} from {start_date}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_latest_signal(symbol, short_window=5, long_window=50):
    """
    Retrieve historical candles from 2024-01-01 to the latest candle, calculate moving averages, 
    and check the latest candle for a trading signal.
    
    Parameters:
        symbol (str): The trading symbol (e.g., "EURUSD").
        short_window (int): The short SMA window size.
        long_window (int): The long SMA window size.
    
    Returns:
        dict: A dictionary with the latest signal and additional information.
    """
    # Initialize MetaTrader 5 connection
    if not mt5.initialize():
        raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")

    df = get_candles_from_start_date(symbol, timeframe="1h", start_date="2024-01-01")

    # Calculate moving averages
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()

    # Generate signals
    df['signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1  # Buy signal
    df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1  # Sell signal
    df['position'] = df['signal'].shift()  # Apply signal on the next bar

    # Get the latest candle's signal
    latest_candle = df.iloc[-1]
    signal_info = {
        'time': latest_candle['time'],
        'close': latest_candle['close'],
        'SMA_short': latest_candle['SMA_short'],
        'SMA_long': latest_candle['SMA_long'],
        'signal': latest_candle['signal'],  # 1: Buy, -1: Sell, 0: No signal
        'position': latest_candle['position']  # Signal applied to the next bar
    }

    # Shutdown MetaTrader 5 connection
    mt5.shutdown()

    return signal_info

# Example usage
# symbol = "GBPNZD"
# signal_info = get_latest_signal(symbol)
# print(f"Latest signal for {symbol}: {signal_info}")

# Set directories
data_directory = "forex_data_pair_per_folder_all_data_1h"
output_directory = "forex_result"
main_function(data_directory, output_directory, days_for_backtest=90)
