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

# Position Sizing Parameters
CAPITAL_ALLOCATION = 0.05  # Allocate 5% of capital per trade
STOP_LOSS_PERCENT = 0.02  # 2% stop-loss
TAKE_PROFIT_PERCENT = 0.05  # 5% take-profit

# ✅ Calculate RSI
def calculate_rsi(data, window=14):
    if 'Close' not in data.columns or len(data) < window:
        print("⚠️ Not enough data to calculate RSI.")
        return pd.Series(np.nan, index=data.index)

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    avg_loss.replace(0, 1e-10, inplace=True)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(0)

# ✅ Fetch Stock & Crypto Data
def fetch_asset_data(symbol):
    """Fetch stock/crypto data and compute indicators."""
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
    """Calculate RSI, SMA, MACD, Bollinger Bands, and ATR."""
    try:
        stock['RSI'] = calculate_rsi(stock)
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
    You are a professional stock and crypto trader providing concise trading recommendations.
    Based on these indicators:
    - RSI: {data.get('RSI', 'N/A')}
    - SMA 50: {data.get('SMA_50', 'N/A')}
    - SMA 200: {data.get('SMA_200', 'N/A')}
    - MACD: {data.get('MACD', 'N/A')}
    - MACD Signal: {data.get('MACD_Signal', 'N/A')}
    - Bollinger Upper: {data.get('Upper_Band', 'N/A')}
    - Bollinger Lower: {data.get('Lower_Band', 'N/A')}
    - ATR: {data.get('ATR', 'N/A')}
    
    Give a concise recommendation: BUY, SELL, or HOLD, with a short reason.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a world expert at stock and crypto trading."},
                {"role": "assistant", "name": "zapier", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"❌ Error querying OpenAI: {e}")
        return "HOLD"

# ✅ Execute Trade with Position Sizing
def execute_trade(symbol, decision):
    try:
        account = alpaca.get_account()
        buying_power = float(account.buying_power)
        trade_amount = buying_power * CAPITAL_ALLOCATION  # 5% allocation
        last_price = float(yf.Ticker(symbol).history(period="1d").iloc[-1]['Close'])
        quantity = int(trade_amount / last_price)

        if decision == "BUY" and quantity > 0:
            alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            print(f"✅ Bought {quantity} shares of {symbol}")
        elif decision == "SELL":
            alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
            print(f"✅ Sold {quantity} shares of {symbol}")
        else:
            print(f"⏸️ Holding position for {symbol}")
    except Exception as e:
        print(f"❌ Error executing trade for {symbol}: {e}")

# ✅ Main Loop
def main():
    while True:
        for asset in ASSETS:
            print(f"📊 Fetching data for {asset}...")
            data = fetch_asset_data(asset)

            if data is not None:
                latest_data = data.iloc[-1].copy().to_dict()
                latest_data = {key[0] if isinstance(key, tuple) else key: value for key, value in latest_data.items()}
                print(f"🔍 Fixed latest_data for {asset}: {latest_data}")

                trade_decision = analyze_with_chatgpt(latest_data)
                print(f"📈 {asset} Decision: {trade_decision}")
                execute_trade(asset, trade_decision)

        print("⏳ Waiting 5 minutes before next check...")
        time.sleep(300)

if __name__ == "__main__":
    main()
