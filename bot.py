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

# Get valid assets from Alpaca
alpaca_assets = {asset.symbol: asset for asset in alpaca.list_assets()}
crypto_assets = {asset.symbol: asset for asset in alpaca.list_assets(asset_class="crypto")}

# Assets to Monitor
ASSETS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "BTC/USD", "ETH/USD"]  # Adjusted crypto symbols for Alpaca

# Position Sizing Parameters
CAPITAL_ALLOCATION = 0.05  # Allocate 5% of capital per trade
STOP_LOSS_PERCENT = 0.02  # 2% stop-loss
TAKE_PROFIT_PERCENT = 0.05  # 5% take-profit
TRADE_LOG_FILE = "trade_log.csv"

# ✅ Calculate RSI
def calculate_rsi(data, window=14):
    if 'Close' not in data.columns or len(data) < window:
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

        return stock.fillna(0)
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

# ✅ Query ChatGPT for Trade Decisions
def analyze_with_chatgpt(data):
    prompt = f"""
    You are a professional stock and crypto trader providing concise trade signals.
    Given these indicators:
    RSI: {data.get('RSI', 'N/A')}, SMA50: {data.get('SMA_50', 'N/A')}, SMA200: {data.get('SMA_200', 'N/A')},
    MACD: {data.get('MACD', 'N/A')}, MACD Signal: {data.get('MACD_Signal', 'N/A')}, 
    Upper Band: {data.get('Upper_Band', 'N/A')}, Lower Band: {data.get('Lower_Band', 'N/A')}.

    Provide a decision: BUY, SELL, or HOLD.
    Format response strictly as:
    "DECISION: [BUY/SELL/HOLD]. REASON: [SHORT REASON]"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a world expert at stock and crypto trading."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"❌ Error querying OpenAI: {e}")
        return "HOLD"

# ✅ Log Trade Actions
def log_trade(symbol, action, quantity, price, reason):
    trade_data = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Action": action,
        "Quantity": quantity,
        "Price": price,
        "Reason": reason
    }
    df = pd.DataFrame([trade_data])
    df.to_csv(TRADE_LOG_FILE, mode='a', header=not os.path.exists(TRADE_LOG_FILE), index=False)
    print(f"📜 Trade logged: {trade_data}")

# ✅ Execute Trade
def execute_trade(symbol, decision, price):
    try:
        is_crypto = symbol in crypto_assets
        if not is_crypto:
            clock = alpaca.get_clock()
            if not clock.is_open:
                print(f"⏸️ Market is closed. Logging trade for {symbol}.")
                log_trade(symbol, "SKIPPED", 0, price, "Market Closed")
                return
        
        account = alpaca.get_account()
        buying_power = float(account.buying_power)
        trade_amount = buying_power * CAPITAL_ALLOCATION
        quantity = round(trade_amount / price, 6)

        reason = decision.split("REASON:")[1].strip() if "REASON:" in decision else decision

        if "BUY" in decision and quantity > 0:
            alpaca.submit_order(symbol=symbol, qty=quantity, side="buy", type="market", time_in_force="gtc")
            print(f"✅ Bought {quantity} of {symbol}")
            log_trade(symbol, "BUY", quantity, price, reason)
        elif "SELL" in decision:
            alpaca.submit_order(symbol=symbol, qty=quantity, side="sell", type="market", time_in_force="gtc")
            print(f"✅ Sold {quantity} of {symbol}")
            log_trade(symbol, "SELL", quantity, price, reason)
        else:
            print(f"⏸️ Holding position for {symbol}")
            log_trade(symbol, "HOLD", 0, price, reason)
    except Exception as e:
        print(f"❌ Error executing trade for {symbol}: {e}")

# ✅ Main Loop
def main():
    while True:
        for asset in ASSETS:
            print(f"📊 Fetching data for {asset}...")
            data = fetch_asset_data(asset)
            if data is not None:
                latest_data = data.iloc[-1].to_dict()
                price = latest_data.get('Close', 0)
                trade_decision = analyze_with_chatgpt(latest_data)
                print(f"📈 {asset} Decision: {trade_decision}")
                execute_trade(asset, trade_decision, price)
        
        print("⏳ Waiting 5 minutes before next check...")
        time.sleep(300)

if __name__ == "__main__":
    main()
