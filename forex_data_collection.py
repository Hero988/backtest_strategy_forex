import numpy as np
import pandas as pd
import os
import MetaTrader5 as mt5
from dotenv import load_dotenv
from datetime import datetime

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

    # Define date ranges
    end_5_years = "2023-12-31"
    start_5_years = "2018-01-01"
    start_present = "2024-01-01"
    end_present = datetime.now().strftime("%Y-%m-%d")

    for symbol in symbols:
        print(f"Collecting data for {symbol}...")

        # Create a folder for the symbol
        symbol_dir = os.path.join(save_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)

        # Gather data for the past 5 years
        df_5_years = gather_forex_data(symbol, timeframe, start_5_years, end_5_years)
        if df_5_years is not None:
            file_path_5_years = os.path.join(symbol_dir, f"{symbol}_5_years.csv")
            df_5_years.to_csv(file_path_5_years)
            print(f"Saved 5-year data for {symbol} to {file_path_5_years}.")

        # Gather data for 2024 to present
        df_present = gather_forex_data(symbol, timeframe, start_present, end_present)
        if df_present is not None:
            file_path_present = os.path.join(symbol_dir, f"{symbol}_2024_present.csv")
            df_present.to_csv(file_path_present)
            print(f"Saved 2024-present data for {symbol} to {file_path_present}.")

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
save_directory = "forex_data_pair_per_folder_1D"

# Collect data for all symbols
collect_and_save_forex_data(symbols, "1d", save_directory)

# Shutdown MetaTrader 5
mt5.shutdown()