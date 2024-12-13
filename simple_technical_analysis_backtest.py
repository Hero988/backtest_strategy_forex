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
    df_copy = df.copy()

    # Generate signals based on RSI
    df_copy['signal'] = 0

    # Detect buy signal (RSI < 30, oversold)
    df_copy.loc[df_copy['RSI'] < 30, 'signal'] = 1

    # Detect sell signal (RSI > 70, overbought)
    df_copy.loc[df_copy['RSI'] > 70, 'signal'] = -1

    # Shift signal to apply on the next bar
    df_copy['position'] = df_copy['signal'].shift()
    return df_copy

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
    stop_loss=100 
    take_profit=200

    df_copy = generate_signals_with_trend(df)
    
    # Backtesting logic
    equity = initial_balance
    max_equity = equity
    point = mt5.symbol_info(pair_folder).point
    position = None
    equity_history = [equity]
    drawdown_history = [0]
    trades = []  # To log trades

    # Adjust lot size for JPY pairs
    lot_size = 100000  # Default lot size
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
        #if equity_history[-1] < initial_balance - 1000:
            #break

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
    Plot the backtesting results using the 'time' column for the x-axis.

    Parameters:
        equity_history (list): Equity balance over time.
        drawdown_history (list): Drawdown history over time.
        df (DataFrame): Updated DataFrame with signals and positions.
        symbol (str): Trading symbol.
        output_dir (str): Directory to save the plot.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Ensure the 'time' column is a datetime object
    if 'time' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['time']):
        raise ValueError("The DataFrame must contain a 'time' column with datetime values.")

    # Plot equity
    aligned_time = df['time'].iloc[:len(equity_history)]
    axs[0].plot(aligned_time, equity_history, label="Equity", color='blue')
    axs[0].set_title(f"Equity Curve ({symbol})")
    axs[0].set_ylabel("Equity")
    axs[0].legend()
    axs[0].grid()

    # Plot drawdown
    axs[1].plot(aligned_time, drawdown_history, label="Drawdown (%)", color='red', linestyle='--')
    axs[1].set_title("Drawdown Over Time")
    axs[1].set_ylabel("Drawdown (%)")
    axs[1].legend()
    axs[1].grid()

    # Plot price and signals
    axs[2].plot(df['time'], df['close'], label="Price", color='black')  # Plot full price series
    buy_signals = df[df['signal'] == 1]['time']
    sell_signals = df[df['signal'] == -1]['time']
    axs[2].scatter(
        buy_signals, 
        df.loc[buy_signals.index, 'close'], 
        label="Buy Signal", 
        marker="^", 
        color="green", 
        alpha=1
    )
    axs[2].scatter(
        sell_signals, 
        df.loc[sell_signals.index, 'close'], 
        label="Sell Signal", 
        marker="v", 
        color="red", 
        alpha=1
    )
    axs[2].set_title("Price and Signals")
    axs[2].set_ylabel("Price")
    axs[2].legend()
    axs[2].grid()

    # Set x-axis label
    plt.xlabel("Time")
    
    # Adjust layout and save the combined plot
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

    backtest_data_segmented = df[df['time'] >= backtest_data_date]

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

def find_earliest_date(symbol, timeframe):
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

    # Ensure the symbol is available in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol {symbol}. Ensure it is in Market Watch.")
        return None

    # Start searching backwards
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 24)  # Start searching 50 years back
    step = timedelta(days=365 * 5)  # Query in chunks of 5 years

    while start_date < end_date:
        rates = mt5.copy_rates_from(symbol, timeframes[timeframe], start_date, 1)
        if rates is not None and len(rates) > 0:
            # Found earliest data
            earliest_date = datetime.utcfromtimestamp(rates[0]['time'])
            return earliest_date
        
        # Move the search window 5 years earlier
        end_date = start_date
        start_date -= step

    print(f"No data available for {symbol} on {timeframe}.")
    return None

# Function to gather Forex data for a single pair
def gather_forex_data(symbol, timeframe, start_date):
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

    end = datetime.now() + timedelta(hours=1)
    rates = mt5.copy_rates_range(symbol, timeframes[timeframe], start_date, end)

    if rates is None:
        print(f"No data retrieved for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_latest_signal(symbol, timeframe='1h'):
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
    
    earliest_date = find_earliest_date(symbol, timeframe)
    df = gather_forex_data(symbol, timeframe="1h", start_date=earliest_date)

    # Save the index as the 'time' column and reset the index
    # Extract the index values and assign them to a new 'time' column
    df['time'] = df.index

    # Reset the index and drop it
    df.reset_index(drop=True, inplace=True)

    # Ensure the 'time' column is in datetime format and set it as the index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    else:
        raise ValueError("The dataset must contain a 'time' column.")

    # Apply the RSI indicator to the DataFrame
    if 'close' in df.columns:
        df['RSI'] = ta.rsi(df['close'], length=14)
    else:
        raise ValueError("The dataset must contain a 'close' column.")
    
    df_with_signals = generate_signals_with_trend(df)
    
    last_10_rows = df_with_signals.tail(10)

    lastest_signals_for_10_rows = generate_signals_with_trend(last_10_rows)

    latest_row = lastest_signals_for_10_rows.tail(1)

    signal_info = latest_row['signal'].iloc[0]

    latest_rsi = latest_row['RSI'].iloc[0]

    latest_time = latest_row['time'].iloc[0]

    # Shutdown MetaTrader 5 connection
    mt5.shutdown()

    return signal_info, latest_row, latest_rsi, latest_time

def execute_trade(symbol, predicted_label, volume, stop_loss_points, take_profit_points):
    """
    Executes a buy or sell order based on the predicted label, with stop loss and take profit.
    Args:
        symbol (str): The symbol to trade.
        predicted_label (str): 'buy' or 'sell'.
        volume (float): The volume of the trade (lot size).
    """
    # Determine action type
    action = mt5.ORDER_TYPE_BUY if predicted_label == 1 else mt5.ORDER_TYPE_SELL

    # Get the latest price
    tick_info = mt5.symbol_info_tick(symbol)
    if tick_info is None:
        print(f"Could not retrieve tick info for {symbol}. Trade aborted.")
        return

    price = tick_info.ask if action == mt5.ORDER_TYPE_BUY else tick_info.bid
    point = mt5.symbol_info(symbol).point  # Symbol point size
    deviation = 20  # Allowed slippage in points

    # Calculate SL and TP
    sl = price - stop_loss_points * point if action == mt5.ORDER_TYPE_BUY else price + stop_loss_points * point
    tp = price + take_profit_points * point if action == mt5.ORDER_TYPE_BUY else price - take_profit_points * point

    # Prepare the trade request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 234000,  # Custom identifier for the trade
        "comment": "Trade executed by script",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or cancel
    }

    # Send the trade request
    result = mt5.order_send(request)
    if result is None:
        print(f"Failed to send trade request for {symbol}.")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to execute {predicted_label} order for {symbol}: {result.retcode}")
        print(f"Error details: {mt5.last_error()}")

def trade_live():
    symbol = "GBPJPY"
    journal_file = "trade_journal.csv"

    while True:
        # Retrieve all open positions
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            # Get the latest signal and row
            signal_info, latest_row, latest_rsi, latest_time = get_latest_signal(symbol)

            if signal_info != 0:
                # Execute the trade
                execute_trade(symbol, predicted_label=signal_info, volume=0.5, stop_loss_points=100, take_profit_points=200)

                # Log the trade in the journal
                trade_data = latest_row.to_dict()
                trade_data['signal'] = signal_info
                trade_data['timestamp'] = datetime.now()

                # Create or append to the trade journal
                if not os.path.exists(journal_file):
                    pd.DataFrame([trade_data]).to_csv(journal_file, index=False)
                else:
                    pd.DataFrame([trade_data]).to_csv(journal_file, mode='a', index=False, header=False)

                print(f"Trade logged in {journal_file}.")

        # Calculate sleep duration until 5 minutes past the next hour
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
        sleep_duration = (next_hour - now).total_seconds()
        
        print(f"Current time: {now}. Sleeping for {sleep_duration // 60:.0f} minutes until {next_hour}, Latest signal for {symbol}: {signal_info}, Latest RSI {latest_rsi}, Latest Signal Time: {latest_time}")
        time.sleep(sleep_duration)
        

def backtest_func():
    # Set directories
    data_directory = "forex_data_pair_per_folder_all_data_1h"
    output_directory = "forex_result"
    main_function(data_directory, output_directory, days_for_backtest=90)

trade_live()
