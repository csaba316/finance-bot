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
    """Calculate RSI using Exponential Moving Average."""
    delta = data['Close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use EMA instead of rolling mean
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    # Prevent division errors by replacing zero avg_loss values
    avg_loss.replace(0, 1e-10, inplace=True)

    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Ensure NaN values are handled correctly
    rsi.fillna(method="bfill", inplace=True)  # Backfill to fill gaps
    rsi.fillna(50, inplace=True)  # Default to 50 (neutral RSI) if still NaN

    return rsi


def fetch_stock_data():
    """ Fetch NVIDIA stock data, resample to 10-minute intervals, and calculate indicators. """
    try:
        stock = yf.download("NVDA", period="7d", interval="10m", group_by="ticker", prepost=True)

        if stock.empty:
            raise ValueError("❌ Yahoo Finance returned an empty dataset. Try increasing the period or changing the interval.")

        # Drop MultiIndex (if present)
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.droplevel(0)  # Remove MultiIndex

        # Ensure required columns exist
        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = expected_columns - set(stock.columns)
        if missing_columns:
            raise ValueError(f"❌ Missing columns in data: {missing_columns}")

        # Resample to 10-minute intervals
        stock = stock.resample('10min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

        # Drop rows where 'Close' is NaN (which happens after resampling)
        stock.dropna(subset=['Close'], inplace=True)

        # Ensure data has no NaN before processing indicators
        stock.fillna(0, inplace=True)

        return stock

    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None


def calculate_indicators(stock):
    """ Calculate RSI, SMA, MACD, Bollinger Bands, and ATR """
    try:
        # Compute RSI
        stock['RSI'] = calculate_rsi(stock)

        # Compute SMA
        stock['SMA_50'] = stock['Close'].rolling(window=50, min_periods=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200, min_periods=200).mean()

        # Compute MACD
        exp12 = stock['Close'].ewm(span=12, adjust=False).mean()
        exp26 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = exp12 - exp26
        stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()

        # Compute Bollinger Bands
        stock['Middle_Band'] = stock['Close'].rolling(window=20).mean()
        stock['Upper_Band'] = stock['Middle_Band'] + (stock['Close'].rolling(window=20).std() * 2)
        stock['Lower_Band'] = stock['Middle_Band'] - (stock['Close'].rolling(window=20).std() * 2)

        # Compute ATR
        stock['High-Low'] = stock['High'] - stock['Low']
        stock['High-Close'] = abs(stock['High'] - stock['Close'].shift())
        stock['Low-Close'] = abs(stock['Low'] - stock['Close'].shift())
        stock['True_Range'] = stock[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        stock['ATR'] = stock['True_Range'].rolling(window=14).mean()

        # Ensure NaN-free data before sending to Zapier
        stock.fillna(0, inplace=True)

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
        if not isinstance(data, dict):
            raise TypeError("❌ Data passed to Zapier is not a dictionary.")

        # Replace None values with 0
        cleaned_data = {key: (0 if value is None else value) for key, value in data.items()}

        print("🚀 Sending Data to Zapier:", cleaned_data)  # Debugging step

        response = requests.post(ZAPIER_WEBHOOK_URL, json=cleaned_data)

        if response.status_code == 200:
            print("✅ Data sent to Zapier successfully!")
        else:
            print("❌ Failed to send data to Zapier. Response:", response.text)

    except Exception as e:
        print(f"❌ Error sending data to Zapier: {e}")


def main():
    """ Main loop to run every 10 minutes """
    while True:
        print("📊 Fetching stock data...")
        stock_data = fetch_stock_data()  # Fetch stock data

        if stock_data is not None and not stock_data.empty:
            stock_data = calculate_indicators(stock_data)

            if stock_data is not None and not stock_data.empty:
                # Debugging RSI & SMA Calculation (Print only if issues persist)
                if stock_data['RSI'].isna().sum() > 0:
                    print("🔍 Debugging RSI Calculation:")
                    print(stock_data[['Close', 'RSI']].tail(20))

                if stock_data['SMA_200'].isna().sum() > 0:
                    print("🔍 Debugging SMA_200 Calculation:")
                    print(stock_data[['Close', 'SMA_200']].tail(20))

                # Get the latest row
                latest_data = stock_data.iloc[-1].fillna(0).to_dict()

                print("✅ Final JSON Payload for Zapier:")
                print(json.dumps(latest_data, indent=2))

                # Send to Zapier
                send_to_zapier(latest_data)

        else:
            print("❌ No valid stock data retrieved. Skipping Zapier request.")

        print("⏳ Waiting 10 minutes for next check...\n")
        time.sleep(600)  # Wait 10 minutes before the next check


if __name__ == "__main__":
    main()
