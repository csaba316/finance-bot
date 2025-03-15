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
ASSETS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]  # Fixed Crypto Symbols

# Position Sizing Parameters
CAPITAL_ALLOCATION = 0.05  # Allocate 5% of capital per trade
STOP_LOSS_PERCENT = 0.02  # 2% stop-loss
TAKE_PROFIT_PERCENT = 0.05  # 5% take-profit
TRADE_LOG_FILE = "trade_log.csv"

# ✅ Fetch Stock Data with Improved Handling
def fetch_asset_data(symbol):
    try:
        interval = "5m" if alpaca.get_clock().is_open else "30m"
        stock = yf.download(symbol, period="7d", interval=interval, auto_adjust=True, prepost=True)

        if stock.empty or stock['Close'].isna().all():
            raise ValueError(f"❌ No valid stock data for {symbol}")

        stock = stock.ffill().dropna()
        stock = calculate_indicators(stock)
        return stock
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

# ✅ Improved Crypto Data Retrieval
def fetch_crypto_data(symbol, retries=3):
    coin_map = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum"}
    coin_id = coin_map.get(symbol, None)

    if not coin_id:
        print(f"❌ Unsupported crypto symbol: {symbol}")
        return None

    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10).json()
            if coin_id not in response or 'usd' not in response[coin_id]:
                raise ValueError(f"❌ No crypto data for {symbol}")
            return response[coin_id]['usd']
        except Exception as e:
            print(f"❌ Attempt {attempt+1} error fetching crypto data for {symbol}: {e}")
            time.sleep(2)
    return None
        
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
    if not data or all(value == 0 or np.isnan(value) for value in data.values()):
        return "DECISION: HOLD. REASON: NOT ENOUGH VALID MARKET DATA."

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
        return "DECISION: HOLD. REASON: API ERROR."
        
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
    df.to_csv("trade_log.csv", mode='a', header=not os.path.exists("trade_log.csv"), index=False)
    print(f"📜 Trade logged: {trade_data}")


# ✅ Execute Trade
def execute_trade(symbol, decision, price):
    try:
        clock = alpaca.get_clock()
        if symbol not in ["BTC-USD", "ETH-USD"] and not clock.is_open:
            print(f"⏸️ Market is closed. Logging trade for {symbol}.")
            log_trade(symbol, "SKIPPED", 0, price, "Market Closed")
            return
        
        account = alpaca.get_account()
        buying_power = float(account.buying_power)
        trade_amount = buying_power * 0.05  # 5% capital allocation
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
            price = 0

            if asset in ["BTC-USD", "ETH-USD"]:
            price = fetch_crypto_data(asset)
            if price:
            print(f"💰 {asset} Price: ${price}")
            latest_data = {'Close': price, 'RSI': np.nan, 'SMA_50': np.nan, 'SMA_200': np.nan,
                       'MACD': np.nan, 'MACD_Signal': np.nan, 'Upper_Band': np.nan, 'Lower_Band': np.nan}
            else:
            print(f"❌ Failed to fetch price for {asset}")
            continue
        
            else:
                stock_data = fetch_asset_data(asset)
                if stock_data is None or stock_data.empty:
                    print(f"❌ No valid data for {asset}. Skipping...")
                    continue
                
                latest_data = stock_data.iloc[-1].to_dict()
                price = latest_data.get('Close', 0)
                if price == 0:
                    print(f"❌ Price data unavailable for {asset}")
                    continue

            trade_decision = analyze_with_chatgpt(latest_data)
            print(f"📈 {asset} Decision: {trade_decision}")
            execute_trade(asset, trade_decision, price)

        print("⏳ Waiting 5 minutes before next check...")
        time.sleep(300)

if __name__ == "__main__":
    main()
