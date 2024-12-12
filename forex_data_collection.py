import numpy as np
import pandas as pd
import os
import MetaTrader5 as mt5
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Load environment variables for MT5 login credentials
load_dotenv()
login = int(os.getenv('MT5_LOGIN'))  # Replace with your login ID
password = os.getenv('MT5_PASSWORD')  # Replace with your password
server = os.getenv('MT5_SERVER')  # Replace with your server name

# Initialize MetaTrader 5 connection
if not mt5.initialize(login=login, password=password, server=server):
    print("Failed to initialize MT5, error code:", mt5.last_error())
    quit()

# Function to gather Forex data for a single pair
def gather_forex_data(symbol, timeframe, start_date, end_date):
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

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    rates = mt5.copy_rates_range(symbol, timeframes[timeframe], start, end)

    if rates is None:
        print(f"No data retrieved for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Function to collect and save data for multiple pairs
def collect_and_save_forex_data(symbols, timeframe, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for symbol in symbols:
        print(f"Collecting data for {symbol}...")

        earliest_date = find_earliest_date(symbol, timeframe)
        if earliest_date is None:
            continue
        start = earliest_date.strftime("%Y-%m-%d")
        end_present = datetime.now().strftime("%Y-%m-%d")

        # Create a folder for the symbol
        symbol_dir = os.path.join(save_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        # Gather data for the past 5 years
        df_all_data = gather_forex_data(symbol, timeframe, start, end_present)
        if df_all_data is not None:
            file_path_all_data = os.path.join(symbol_dir, f"{symbol}_all_data.csv")
            df_all_data.to_csv(file_path_all_data)
            print(f"Saved All data for {symbol} to {file_path_all_data}.")

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

# List of currency pairs to collect
symbols = [
    "EURUSD", "GBPCAD", "GBPNZD", "AUDCAD", "GBPUSD", 
    "AUDUSD", "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY",
    "NZDUSD", "CHFJPY", "EURGBP", "EURAUD", "EURCHF",
    "EURJPY", "EURCAD", "GBPCHF", "GBPJPY", "USDCAD",
    "CADCHF", "CADJPY", "GBPAUD", "USDCHF", "USDJPY",
    "NZDCAD", "NZDCHF", "NZDJPY", 
]

# Directory to save CSV files
save_directory = "forex_data_pair_per_folder_all_data_1h"

# Collect data for all symbols
collect_and_save_forex_data(symbols, "1h", save_directory)

# Shutdown MetaTrader 5
mt5.shutdown()
