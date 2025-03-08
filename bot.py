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
    """ Fetch NVIDIA stock data & compute indicators. """
    try:
        stock = yf.download("NVDA", period="7d", interval="5m", group_by="ticker", prepost=True)

        if stock.empty:
            raise ValueError("❌ Yahoo Finance returned an empty dataset.")

        stock = stock[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Resample to 5-minute intervals
        stock = stock.resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        stock['RSI'] = calculate_rsi(stock)
        stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200).mean()

        # MACD
        stock['MACD'] = stock['Close'].ewm(span=12, adjust=False).mean() - stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        stock['Middle_Band'] = stock['Close'].rolling(window=20).mean()
        stock['Upper_Band'] = stock['Middle_Band'] + (stock['Close'].rolling(window=20).std() * 2)
        stock['Lower_Band'] = stock['Middle_Band'] - (stock['Close'].rolling(window=20).std() * 2)

        # ATR (Volatility Measurement)
        stock['High-Low'] = stock['High'] - stock['Low']
        stock['High-Close'] = abs(stock['High'] - stock['Close'].shift())
        stock['Low-Close'] = abs(stock['Low'] - stock['Close'].shift())
        stock['True_Range'] = stock[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        stock['ATR'] = stock['True_Range'].rolling(window=14).mean()

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
    """ Main loop to run every 5 minutes. """
    while True:
        print("📊 Fetching stock data...")
        stock_data = fetch_stock_data()

        if stock_data is not None and not stock_data.empty:
            latest_data = stock_data.iloc[-1].to_dict()
            send_to_zapier(latest_data)

        print("⏳ Waiting 5 minutes for next check...\n")
        time.sleep(300)  # 5-minute wait

if __name__ == "__main__":
    main()
