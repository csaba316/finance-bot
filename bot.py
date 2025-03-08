import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# Environment variables
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")

def fetch_stock_data():
    """ Fetch NVIDIA stock data, resample to 5-minute intervals, and calculate indicators. """
    try:
        stock = yf.download("NVDA", period="7d", interval="5m", group_by="ticker", prepost=True)

        if stock.empty:
            raise ValueError("❌ Yahoo Finance returned an empty dataset. Try increasing the period or changing the interval.")
        
        # Drop MultiIndex if present
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.droplevel(0)

        print("✅ Fixed Columns:", stock.columns)  # Debugging Step

        # Ensure required columns exist
        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = expected_columns - set(stock.columns)
        if missing_columns:
            raise ValueError(f"❌ Missing columns in data: {missing_columns}")

        # Resample to 5-minute intervals
        stock = stock.resample('5min').agg({
            'Open': 'first', 
            'High': 'max', 
            'Low': 'min', 
            'Close': 'last', 
            'Volume': 'sum'
        })

        # Drop rows where 'Close' is NaN (which happens after resampling)
        stock.dropna(subset=['Close'], inplace=True)

        print("📊 Checking NaN values before RSI calculation:")
        print(stock[['Close']].isna().sum())  # Should return 0

        # Calculate RSI
        stock['RSI'] = calculate_rsi(stock)

        # Drop initial NaN values and fill the rest
        stock['RSI'] = stock['RSI'].dropna()
        stock['RSI'].fillna(50, inplace=True)  

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

def calculate_rsi(data, window=14):
    """ Calculate RSI using Exponential Moving Average. """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, 1e-10))  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)  # Default to 50 (neutral RSI)

def determine_trade_signal(stock):
    """ Determine trade signal based on RSI & MACD. """
    latest = stock.iloc[-1]

    if latest['RSI'] < 30 and latest['MACD'] > latest['MACD_Signal']:
        return "BUY"
    elif latest['RSI'] > 70 and latest['MACD'] < latest['MACD_Signal']:
        return "SELL"
    else:
        return "HOLD"

def send_to_zapier(data):
    """ Send latest stock indicators to Zapier webhook. """
    if not ZAPIER_WEBHOOK_URL:
        print("❌ Error: Zapier Webhook URL is not set.")
        return

    try:
        payload = {
            "Stock": "NVDA",
            "Open": data["Open"],
            "High": data["High"],
            "Low": data["Low"],
            "Close": data["Close"],
            "Volume": data["Volume"],
            "RSI": data["RSI"],
            "SMA_50": data["SMA_50"],
            "SMA_200": data["SMA_200"],
            "MACD": data["MACD"],
            "MACD_Signal": data["MACD_Signal"],
            "Upper_Band": data["Upper_Band"],
            "Lower_Band": data["Lower_Band"],
            "ATR": data["ATR"],
            "Trade_Signal": determine_trade_signal(data)
        }

        print("🚀 Sending Data to Zapier:", json.dumps(payload, indent=2))
        response = requests.post(ZAPIER_WEBHOOK_URL, json=payload)

        if response.status_code == 200:
            print("✅ Data sent to Zapier successfully!")
        else:
            print("❌ Failed to send data to Zapier. Response:", response.text)

    except Exception as e:
        print(f"❌ Error sending data to Zapier: {e}")

def main():
    """ Main loop to run every 10 minutes. """
    while True:
        print("📊 Fetching stock data...")
        stock_data = fetch_stock_data()

        if stock_data is not None and not stock_data.empty:
            latest_data = stock_data.iloc[-1].to_dict()
            send_to_zapier(latest_data)

        print("⏳ Waiting 10 minutes for next check...\n")
        time.sleep(600)  # 10-minute wait

if __name__ == "__main__":
    main()
