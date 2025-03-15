import os
import time
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
from alpaca_trade_api import REST

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Initialize APIs
client = OpenAI(api_key=OPENAI_API_KEY)
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# Assets to Monitor
ASSETS = ["NVDA", "AAPL", "TSLA", "BTC-USD", "ETH-USD"]

# ✅ Calculate RSI
def calculate_rsi(data, window=14):
    if 'Close' not in data.columns or len(data) < window:
        print("⚠️ Not enough data to calculate RSI.")
        return pd.Series(np.nan, index=data.index)  # Return NaN series to avoid KeyError
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    
    avg_loss.replace(0, 1e-10, inplace=True)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    print("✅ RSI Calculated: ", rsi.tail())
    return rsi.fillna(0)

# ✅ Fetch Stock & Crypto Data
def fetch_asset_data(symbol):
    try:
        stock = yf.download(symbol, period="7d", interval="5m", auto_adjust=False, prepost=True)
        if stock.empty:
            raise ValueError(f"❌ No data for {symbol}")
        
        stock = calculate_indicators(stock)
        return stock
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

# ✅ Calculate Indicators
def calculate_indicators(stock):
    try:
        stock['RSI'] = calculate_rsi(stock)

        if stock['RSI'].isna().all():
            print("⚠️ Warning: RSI is completely NaN for this asset.")

        stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200).mean()

        exp12 = stock['Close'].ewm(span=12, adjust=False).mean()
        exp26 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = exp12 - exp26
        stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()

        stock['Middle_Band'] = stock['Close'].rolling(window=20).mean()
        stock['Std_Dev'] = stock['Close'].rolling(window=20).std()
        stock['Upper_Band'] = stock['Middle_Band'] + (stock['Std_Dev'] * 2)
        stock['Lower_Band'] = stock['Middle_Band'] - (stock['Std_Dev'] * 2)

        stock['High-Low'] = stock['High'] - stock['Low']
        stock['High-Close'] = abs(stock['High'] - stock['Close'].shift())
        stock['Low-Close'] = abs(stock['Low'] - stock['Close'].shift())
        stock['True_Range'] = stock[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        stock['ATR'] = stock['True_Range'].rolling(window=14).mean()

        return stock.fillna(0)
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

# ✅ Query ChatGPT for Trade Decisions
def analyze_with_chatgpt(data):
    prompt = f"""
    Given the following stock indicators:
    - RSI: {data.get('RSI', 'N/A')}
    - SMA 50: {data.get('SMA_50', 'N/A')}
    - SMA 200: {data.get('SMA_200', 'N/A')}
    - MACD: {data.get('MACD', 'N/A')}
    - MACD Signal: {data.get('MACD_Signal', 'N/A')}
    - Bollinger Upper: {data.get('Upper_Band', 'N/A')}
    - Bollinger Lower: {data.get('Lower_Band', 'N/A')}
    - ATR: {data.get('ATR', 'N/A')}
    Should I BUY, SELL, or HOLD?
    """
    response = client.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message['content'].strip().upper()

# ✅ Execute Trade on Alpaca
def execute_trade(symbol, decision):
    if decision == "BUY":
        alpaca.submit_order(symbol=symbol, qty=1, side="buy", type="market", time_in_force="gtc")
        print(f"✅ Bought 1 share of {symbol}")
    elif decision == "SELL":
        alpaca.submit_order(symbol=symbol, qty=1, side="sell", type="market", time_in_force="gtc")
        print(f"✅ Sold 1 share of {symbol}")
    else:
        print(f"⏸️ Holding position for {symbol}")

# ✅ Main Loop
def main():
    while True:
        for asset in ASSETS:
            print(f"📊 Fetching data for {asset}...")
            data = fetch_asset_data(asset)
            
            if data is not None:
                latest_data = data.iloc[-1].to_dict()
                if "RSI" not in latest_data:
                    print(f"⚠️ Warning: RSI missing from latest_data for {asset}. Debug info: \n", data.tail())
                trade_decision = analyze_with_chatgpt(latest_data)
                print(f"📈 {asset} Decision: {trade_decision}")
                execute_trade(asset, trade_decision)
        
        print("⏳ Waiting 5 minutes before next check...")
        time.sleep(300)

if __name__ == "__main__":
    main()
