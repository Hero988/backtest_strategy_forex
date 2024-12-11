import os
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from dotenv import load_dotenv

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

def backtest_strategy(df, pair_folder, pair_output_dir, initial_balance=10000, stop_loss=50, take_profit=100):
    """
    Backtest a simple SMA crossover strategy with stop loss and take profit.

    Parameters:
        df (DataFrame): Historical OHLC data.
        pair_folder (str): Folder to save trade logs and statistics.
        initial_balance (float): Starting balance for the backtest.
        stop_loss (float): Stop loss in pips.
        take_profit (float): Take profit in pips.

    Returns:
        equity_history (list): Equity balance over time.
        drawdown_history (list): Drawdown history over time.
        df (DataFrame): Updated DataFrame with signals and positions.
    """
    short_window = 5
    long_window = 50
    df_copy = df.copy()

    # Calculate moving averages
    df_copy['SMA_short'] = df_copy['close'].rolling(window=short_window).mean()
    df_copy['SMA_long'] = df_copy['close'].rolling(window=long_window).mean()

    # Generate signals
    df_copy['signal'] = 0
    df_copy.loc[df_copy['SMA_short'] > df_copy['SMA_long'], 'signal'] = 1  # Buy signal
    df_copy.loc[df_copy['SMA_short'] < df_copy['SMA_long'], 'signal'] = -1  # Sell signal
    df_copy['position'] = df_copy['signal'].shift()  # Apply signal on the next bar

    # Adjust lot size for JPY pairs
    lot_size = 10000  # Default lot size
    if "JPY" in pair_folder:
        lot_size = 100  # Adjust to a smaller, more reasonable lot size for JPY pairs

    # Backtesting logic
    equity = initial_balance
    max_equity = equity
    point = mt5.symbol_info(pair_folder).point
    position = None
    equity_history = [equity]
    drawdown_history = [0]
    trades = []  # To log trades

    for i in range(1, len(df_copy)):
        row = df_copy.iloc[i]

        # Check for an active position
        if position is not None:
            if position['type'] == 'sell':
                # Check take profit for sell
                if row['low'] <= position['take_profit']:
                    profit = (position['entry'] - position['take_profit']) * lot_size
                    equity += profit
                    trades.append({'Type': 'Sell', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': profit, 'Entry': position['entry'],
                                   'Exit': position['take_profit'], 'Result': 'Win', 'Equity': equity})
                    position = None  # Close the position
                # Check stop loss for sell
                elif row['high'] >= position['stop_loss']:
                    loss = (position['stop_loss'] - position['entry']) * lot_size
                    equity -= loss
                    trades.append({'Type': 'Sell', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': -loss, 'Entry': position['entry'],
                                   'Exit': position['stop_loss'], 'Result': 'Loss', 'Equity': equity})
                    position = None  # Close the position

            elif position['type'] == 'buy':
                # Check take profit for buy
                if row['high'] >= position['take_profit']:
                    profit = (position['take_profit'] - position['entry']) * lot_size
                    equity += profit
                    trades.append({'Type': 'Buy', 'Entry Time': position['entry_time'], 'Exit Time': row['time'], 'Profit/Loss': profit, 'Entry': position['entry'],
                                   'Exit': position['take_profit'], 'Result': 'Win', 'Equity': equity})
                    position = None  # Close the position
                # Check stop loss for buy
                elif row['low'] <= position['stop_loss']:
                    loss = (position['entry'] - position['stop_loss']) * lot_size
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

    return equity_history, drawdown_history, df_copy

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



def main_function(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the main output directory exists

    for pair_folder in os.listdir(data_dir):
        pair_path = os.path.join(data_dir, pair_folder)
        if not os.path.isdir(pair_path):
            continue

        # Create a subfolder inside output_dir for the pair_folder
        pair_output_dir = os.path.join(output_dir, pair_folder)
        os.makedirs(pair_output_dir, exist_ok=True)

        # Load Data
        five_years_data_path = os.path.join(pair_path, f"{pair_folder}_5_years.csv")
        one_year_data_path = os.path.join(pair_path, f"{pair_folder}_2024_present.csv")
        if not os.path.exists(five_years_data_path) or not os.path.exists(one_year_data_path):
            print(f"Data files missing for {pair_folder}")
            continue

        df_five_years = pd.read_csv(five_years_data_path)
        df_1_year = pd.read_csv(one_year_data_path)

        equity_history, drawdown_history, df_copy = backtest_strategy(df_1_year,pair_folder, pair_output_dir, initial_balance=10000, stop_loss=50, take_profit=100)
        plot_results(equity_history, drawdown_history, df_copy, pair_folder, pair_output_dir)

        print(f"Completed backtest {pair_folder}")


# Set directories
data_directory = "forex_data_pair_per_folder"
output_directory = "forex_result"
main_function(data_directory, output_directory)