import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# Replace with your Zapier Webhook URL
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")

def calculate_rsi(data, window=14):
    """Calculate RSI using 10-minute closing prices."""
    delta = data['Close'].diff()  # Get price difference between periods

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)  
    loss = -delta.where(delta < 0, 0)  

    # Calculate rolling averages for gains and losses
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Calculate RSI
    rs = avg_gain / (avg_loss + 1e-10)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

    
def fetch_stock_data():
    """ Fetch NVIDIA stock data, resample to 10-minute intervals, and calculate indicators. """
    try:
        stock = yf.download("NVDA", period="5d", interval="1m", group_by="ticker")

        if stock.empty:
            raise ValueError("❌ Yahoo Finance returned an empty dataset. Try increasing the period or changing the interval.")

        # Drop MultiIndex (if present)
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.droplevel(0)  # Remove MultiIndex

        print("✅ Fixed Columns:", stock.columns)  # Debugging Step

        # Ensure required columns exist
        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = expected_columns - set(stock.columns)
        if missing_columns:
            raise ValueError(f"❌ Missing columns in data: {missing_columns}")

        # Resample to 10-minute intervals
        stock = stock.resample('10min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

        # Drop rows where 'Close' is NaN (which happens after resampling)
        stock.dropna(subset=['Close'], inplace=True)

        # Debugging Step: Ensure Close column is valid before RSI calculation
        print("📊 Checking NaN values before RSI calculation:")
        print(stock[['Close']].isna().sum())  # Should return 0

        # Calculate RSI
        stock['RSI'] = calculate_rsi(stock)

        # Fix remaining NaN values in RSI
        stock['RSI'] = stock['RSI'].bfill()  # Backfill NaN values
        stock['RSI'] = stock['RSI'].fillna(50)  # Default to 50 if still NaN (neutral RSI)

        # Debugging Step: Check if RSI column is NaN after calculation
        print("📈 Checking NaN values after RSI calculation:")
        print(stock[['RSI']].isna().sum())  # Should return 0

        # Compute SMA
        stock['SMA_50'] = stock['Close'].rolling(window=50, min_periods=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200, min_periods=200).mean()

        # Fill remaining NaN values
        stock.fillna(0, inplace=True)

        return stock
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None

        
def calculate_indicators(stock):
    """ Calculate RSI, SMA, MACD, Bollinger Bands, and ATR """
    try:
        # RSI Calculation
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff(1)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
            avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # Moving Averages
        stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200).mean()

        # MACD Calculation
        exp12 = stock['Close'].ewm(span=12, adjust=False).mean()
        exp26 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = exp12 - exp26
        stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        stock['Middle_Band'] = stock['Close'].rolling(window=20).mean()
        stock['Upper_Band'] = stock['Middle_Band'] + (stock['Close'].rolling(window=20).std() * 2)
        stock['Lower_Band'] = stock['Middle_Band'] - (stock['Close'].rolling(window=20).std() * 2)

        # Average True Range (ATR)
        stock['High-Low'] = stock['High'] - stock['Low']
        stock['High-Close'] = abs(stock['High'] - stock['Close'].shift())
        stock['Low-Close'] = abs(stock['Low'] - stock['Close'].shift())
        stock['True_Range'] = stock[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        stock['ATR'] = stock['True_Range'].rolling(window=14).mean()

        # RSI Calculation
        stock['RSI'] = calculate_rsi(stock)

        return stock
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

def send_to_zapier(data):
    """ Send latest stock indicators to Zapier webhook """
    if not ZAPIER_WEBHOOK_URL:
        print("❌ Error: Zapier Webhook URL is not set.")
        return

    try:
        # Replace NaN values with 0 or "N/A"
        data = data.fillna(0)  # Converts NaN to 0 (or use "N/A" if preferred)
        
        payload = {
            "Stock": "NVDA",
            "Close_Price": data['Close'],
            "RSI": data['RSI'],
            "SMA_50": data['SMA_50'],
            "SMA_200": data['SMA_200'],
            "MACD": data['MACD'],
            "MACD_Signal": data['MACD_Signal'],
            "Upper_Band": data['Upper_Band'],
            "Lower_Band": data['Lower_Band'],
            "ATR": data['ATR']
        }

        response = requests.post(ZAPIER_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            print("✅ Data sent to Zapier successfully!")
        else:
            print("❌ Failed to send data to Zapier. Response:", response.text)
    except Exception as e:
        print(f"❌ Error sending data to Zapier: {e}")

import time

def main():
    """ Main loop to run every 10 minutes """
    while True:
        print("📊 Fetching stock data...")
        stock_data = fetch_stock_data()  # Fetch stock data

        if stock_data is not None and not stock_data.empty:  # Ensure stock data is valid
            stock_data = calculate_indicators(stock_data)  # Apply indicators

            if stock_data is not None and not stock_data.empty:  # Re-check after indicators
                # Get the latest row including RSI correctly
                latest_data = stock_data.iloc[[-1]].copy()  # Extract the last row as a DataFrame

                print("📈 Latest Stock Data:")
                print(latest_data)

                # Convert DataFrame row to a dictionary properly
                json_payload = latest_data.reset_index().to_dict(orient="records")[0]  # Reset index to fix errors

                # Send to Zapier
                send_to_zapier(json_payload)

        else:
            print("❌ No valid stock data retrieved. Skipping Zapier request.")

        print("⏳ Waiting 10 minutes for next check...\n")
        time.sleep(600)  # Wait 10 minutes before the next check

if __name__ == "__main__":
    main()
