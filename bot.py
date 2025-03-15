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

# ✅ Fetch Stock & Crypto Data
def fetch_asset_data(symbol):
    """Fetch stock/crypto data and compute indicators."""
    try:
        stock = yf.download(symbol, period="7d", interval="5m", prepost=True)
        if stock.empty:
            raise ValueError(f"❌ No data for {symbol}")
        
        stock['RSI'] = calculate_rsi(stock)
        stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200).mean()
        
        return stock
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

# ✅ Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    
    avg_loss.replace(0, 1e-10, inplace=True)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ✅ Query ChatGPT for Trade Decisions
def analyze_with_chatgpt(data):
    """Send market indicators to ChatGPT for analysis."""
    prompt = f"""
    Given the following stock indicators:
    - RSI: {data['RSI']}
    - SMA 50: {data['SMA_50']}
    - SMA 200: {data['SMA_200']}
    Should I BUY, SELL, or HOLD?
    """
    response = client.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message['content'].strip().upper()

# ✅ Execute Trade on Alpaca
def execute_trade(symbol, decision):
    """Place a trade order based on ChatGPT's decision."""
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
    """Monitor assets and trade based on AI signals."""
    while True:
        for asset in ASSETS:
            print(f"📊 Fetching data for {asset}...")
            data = fetch_asset_data(asset)
            
            if data is not None:
                latest_data = data.iloc[-1].to_dict()
                trade_decision = analyze_with_chatgpt(latest_data)
                print(f"📈 {asset} Decision: {trade_decision}")
                execute_trade(asset, trade_decision)
        
        print("⏳ Waiting 5 minutes before next check...")
        time.sleep(300)

if __name__ == "__main__":
    main()
