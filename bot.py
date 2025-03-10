import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# Load Zapier Webhook URL
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")

# File to store last valid stock data
LAST_VALID_DATA_FILE = "last_valid_data.json"

# ✅ **Persist Last Valid Data**
def store_last_valid_data(data):
    """Store last valid stock data to a JSON file."""
    with open(LAST_VALID_DATA_FILE, "w") as file:
        json.dump(data, file)

def load_last_valid_data():
    """Load last valid stock data from a JSON file."""
    if os.path.exists(LAST_VALID_DATA_FILE):
        with open(LAST_VALID_DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# ✅ **Detect If Market is Open**
def is_market_open(stock_data):
    """Check if the market is open based on trading volume."""
    return stock_data["Volume"].iloc[-1] > 0  # Market open if volume > 0

# ✅ **Calculate RSI**
def calculate_rsi(data, window=14):
    """Calculate RSI using EMA, reusing last stored RSI if market is closed."""
    
    # Load last stored RSI
    last_data = load_last_valid_data()
    
    # If the market is closed, use last stored RSI but still return a Series
    if not is_market_open(data):
        if last_data and "RSI" in last_data:
            print("⏸️ Market is closed. Using last stored RSI.")
            return pd.Series(last_data["RSI"], index=data.index)  # Return last RSI as a Series
    
    # Compute RSI normally
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    avg_loss.replace(0, 1e-10, inplace=True)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Store last valid RSI
    store_last_valid_data({"RSI": rsi.iloc[-1]})
    
    return rsi

# ✅ **Fetch Stock Data**
def fetch_stock_data():
    """Fetch NVIDIA stock data, resample to 5-minute intervals, and calculate indicators."""
    try:
        stock = yf.download("NVDA", period="7d", interval="5m", group_by="ticker", prepost=True)

        if stock.empty:
            raise ValueError("❌ Yahoo Finance returned an empty dataset. Try increasing the period or changing the interval.")
        
        # Drop MultiIndex if present
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.droplevel(0)

        print("✅ Fixed Columns:", stock.columns)

        # Ensure required columns exist
        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_columns = expected_columns - set(stock.columns)
        if missing_columns:
            raise ValueError(f"❌ Missing columns in data: {missing_columns}")

        # Resample to 10-minute intervals
        stock = stock.resample('5min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        stock.dropna(subset=['Close'], inplace=True)  # Drop NaN close values

        # Calculate RSI
        stock['RSI'] = calculate_rsi(stock)

        # Compute SMA
        stock['SMA_50'] = stock['Close'].rolling(window=50, min_periods=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200, min_periods=200).mean()

        # Fill remaining NaN values
        stock.fillna(0, inplace=True)

        return stock
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return None

# ✅ **Calculate Other Indicators**
def calculate_indicators(stock):
    """Calculate RSI, SMA, MACD, Bollinger Bands, and ATR."""
    try:
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

        # ATR Calculation
        stock['High-Low'] = stock['High'] - stock['Low']
        stock['High-Close'] = abs(stock['High'] - stock['Close'].shift())
        stock['Low-Close'] = abs(stock['Low'] - stock['Close'].shift())
        stock['True_Range'] = stock[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        stock['ATR'] = stock['True_Range'].rolling(window=14).mean()

        return stock
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

def send_to_zapier(data):
    """Send stock data to Zapier webhook."""
    if not ZAPIER_WEBHOOK_URL:
        print("❌ Error: Zapier Webhook URL is not set.")
        return

    try:
        response = requests.post(ZAPIER_WEBHOOK_URL, json=data)
        if response.status_code == 200:
            print("✅ Data sent to Zapier successfully!")
        else:
            print("❌ Failed to send data to Zapier. Response:", response.text)
    except Exception as e:
        print(f"❌ Error sending data to Zapier: {e}")

# ✅ **Main Loop**
def main():
    """Main loop to run every 5 minutes."""
    while True:
        print("📊 Fetching stock data...")
        stock_data = fetch_stock_data()

        if stock_data is not None and not stock_data.empty:
            stock_data = calculate_indicators(stock_data)

            if stock_data is not None and not stock_data.empty:
                latest_data = stock_data.iloc[[-1]].copy()
                json_payload = latest_data.fillna(0).reset_index().to_dict(orient="records")[0]

                # ✅ Convert Timestamp to string
                if "Datetime" in json_payload:
                    json_payload["Datetime"] = str(json_payload["Datetime"])

                print("📈 Latest Stock Data (Always Visible):")
                print(json.dumps(json_payload, indent=2))

                if not is_market_open(stock_data):
                    print("⏸️ Market is closed. Skipping Zapier request.")
                else:
                    print("🚀 Sending Data to Zapier:", json_payload)
                    send_to_zapier(json_payload)

        print("⏳ Waiting 5 minutes for next check...\n")
        time.sleep(300)

if __name__ == "__main__":
    main()
