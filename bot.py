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
ZAPIER_ASSISTANT_ID = os.getenv("ZAPIER_ASSISTANT_ID")

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
        clock = alpaca.get_clock()
        interval = "5m" if clock.is_open else "30m"

        # Attempt fetching intraday data
        stock = yf.download(symbol, period="7d", interval=interval, auto_adjust=True, prepost=True)

        # ✅ Handle Multi-Index Issue
        if isinstance(stock.columns, pd.MultiIndex):
            stock = stock.xs(symbol, level=1, axis=1)

        # ✅ Debug: Print fetched data
        if stock.empty:
            print(f"⚠️ No data for {symbol} using interval {interval}. Trying daily data...")
            stock = yf.download(symbol, period="30d", interval="1d", auto_adjust=True)

            if isinstance(stock.columns, pd.MultiIndex):
                stock = stock.xs(symbol, level=1, axis=1)

        if stock.empty:
            raise ValueError(f"❌ No valid stock data for {symbol} (empty DataFrame)")

        # ✅ Ensure 'Close' column exists and has valid values
        if 'Close' not in stock.columns:
            raise ValueError(f"❌ 'Close' price missing for {symbol} (columns: {stock.columns.tolist()})")

        # ✅ Debug: Print last few rows of stock data
        print(f"📊 {symbol} data preview:\n{stock[['Close']].tail(3)}")

        # Forward-fill missing values and calculate indicators
        stock = stock.ffill().dropna()
        stock = calculate_indicators(stock)
        return stock

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

# ✅ Improved Crypto Data Retrieval
def fetch_crypto_data(symbol):
    yahoo_symbol_map = {
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD"
    }
    yahoo_symbol = yahoo_symbol_map.get(symbol, symbol)  # Ensure correct symbol format

    try:
        # ✅ Fetch last 7 days of data with 1-hour intervals
        crypto_data = yf.download(yahoo_symbol, period="7d", interval="1h", auto_adjust=True, prepost=True)

        # ✅ Properly check if data is empty
        if crypto_data.empty:
            raise ValueError(f"❌ No valid crypto data for {symbol}")

        # ✅ Forward-fill missing values and drop remaining NaN rows
        crypto_data = crypto_data.ffill().dropna()

        return calculate_indicators(crypto_data)

    except Exception as e:
        print(f"❌ Error fetching Yahoo Finance data for {symbol}: {e}")
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
        # ✅ Calculate RSI
        stock['RSI'] = calculate_rsi(stock)

        # ✅ Moving Averages
        stock['SMA_50'] = stock['Close'].rolling(window=50).mean()
        stock['SMA_200'] = stock['Close'].rolling(window=200).mean()

        # ✅ Exponential Moving Averages (EMA)
        stock['EMA_9'] = stock['Close'].ewm(span=9, adjust=False).mean()
        stock['EMA_21'] = stock['Close'].ewm(span=21, adjust=False).mean()

        # ✅ MACD Indicator
        exp12 = stock['Close'].ewm(span=12, adjust=False).mean()
        exp26 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = exp12 - exp26
        stock['MACD_Signal'] = stock['MACD'].ewm(span=9, adjust=False).mean()

        # ✅ Bollinger Bands
        stock['Middle_Band'] = stock['Close'].rolling(window=20).mean()
        stock['Std_Dev'] = stock['Close'].rolling(window=20).std()
        stock['Upper_Band'] = stock['Middle_Band'] + (stock['Std_Dev'] * 2)
        stock['Lower_Band'] = stock['Middle_Band'] - (stock['Std_Dev'] * 2)

        # ✅ VWAP Calculation
        stock['VWAP'] = (stock['Close'] * stock['Volume']).cumsum() / stock['Volume'].cumsum()

        # ✅ Ensure no NaNs are present
        return stock.fillna(0)

    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

# ✅ Query ChatGPT for Trade Decisions
def analyze_with_chatgpt(data):
    if not data or all(value == 0 or np.isnan(value) for value in data.values()):
        return "DECISION: HOLD. REASON: NOT ENOUGH VALID MARKET DATA."

    prompt = f"""
    You are an expert stock and crypto trader. Analyze these indicators:

    RSI: {data.get('RSI', 'N/A')}, 
    SMA50: {data.get('SMA_50', 'N/A')}, SMA200: {data.get('SMA_200', 'N/A')},
    EMA9: {data.get('EMA_9', 'N/A')}, EMA21: {data.get('EMA_21', 'N/A')},
    MACD: {data.get('MACD', 'N/A')}, MACD Signal: {data.get('MACD_Signal', 'N/A')},
    VWAP: {data.get('VWAP', 'N/A')},
    Upper Band: {data.get('Upper_Band', 'N/A')}, Lower Band: {data.get('Lower_Band', 'N/A')}.

    **Revised Trade Strategy:**
    - **BUY when:**  
      - EMA9 **crosses above** EMA21 **AND** MACD is **above the signal line**, OR  
      - RSI is **between 50-65** AND price is **above VWAP** AND **near Lower Band**.  
      - If 2 or more of these conditions align, it’s a strong BUY signal.  

    - **SELL when:**  
      - EMA9 **crosses below** EMA21 **AND** MACD is **below the signal line**, OR  
      - RSI is **above 70** (overbought) **AND** price is **near Upper Band**.  
      - If 2 or more of these conditions align, it’s a strong SELL signal.  

    - **HOLD only if:** Indicators conflict or there is **NO strong buy/sell signal**.

    Format response strictly as:
    "DECISION: [BUY/SELL/HOLD]. REASON: [SHORT EXPLANATION]"
    """

    try:
        # ✅ Create a new thread for conversation
        thread = client.beta.threads.create()

        # ✅ Add user message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        # ✅ Run the assistant on that thread
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ZAPIER_ASSISTANT_ID
        )

        # ✅ Wait for the response to be generated
        while run.status in ["queued", "in_progress"]:
            time.sleep(2)  # Wait for processing
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        # ✅ Fetch the response
        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value.strip().upper()
        else:
            raise ValueError(f"Unexpected run status: {run.status}")

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

def queue_trade(symbol, decision, price, reason):
    trade_data = {
        "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Action": "QUEUED",
        "Quantity": 0,
        "Price": price,
        "Reason": reason
    }
    df = pd.DataFrame([trade_data])
    df.to_csv("queued_trades.csv", mode='a', header=not os.path.exists("queued_trades.csv"), index=False)
    print(f"📋 Trade queued: {trade_data}")

# ✅ Execute Trade
def execute_trade(symbol, decision, price, reason):
    """Executes a trade based on the decision from ChatGPT."""
    try:
        if price <= 0:
            print(f"❌ Invalid price for {symbol}. Skipping trade...")
            return

        # ✅ Fetch Alpaca account balance
        account = alpaca.get_account()
        buying_power = float(account.buying_power)

        if "BUY" in decision:
            # ✅ Calculate trade amount (5% of available capital)
            trade_amount = min(buying_power * CAPITAL_ALLOCATION, float(account.cash))

            if trade_amount < 10:
                print(f"❌ Trade amount for {symbol} is below Alpaca's minimum ($10). Skipping trade...")
                return

            # ✅ Calculate quantity (rounded for fractional shares & crypto)
            quantity = round(trade_amount / price, 6)

            if quantity <= 0:
                print(f"❌ Trade quantity too small for {symbol}. Skipping trade...")
                return

            # ✅ Force `time_in_force="day"` for fractional orders
            time_in_force = "day" if quantity < 1 else "gtc"

            # ✅ Execute trade
            try:
                alpaca.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side="buy",
                    type="market",
                    time_in_force=time_in_force
                )
                print(f"✅ Bought {quantity} of {symbol} at ${price:.2f} (Order Type: {time_in_force})")
                log_trade(symbol, "BUY", quantity, price, reason)
            except Exception as e:
                print(f"❌ Error executing trade for {symbol}: {e}")

        elif "SELL" in decision:
            try:
                position = alpaca.get_position(symbol)
                available_qty = float(position.qty)

                if available_qty <= 0:
                    print(f"❌ No available shares of {symbol} to sell. Skipping trade...")
                    return

                # ✅ Sell entire position
                quantity = available_qty

                if quantity * price < 10:
                    print(f"❌ Trade amount for {symbol} is below Alpaca's minimum ($10). Skipping trade...")
                    return

                # ✅ Force `time_in_force="day"` for fractional orders
                time_in_force = "day" if quantity < 1 else "gtc"

                # ✅ Execute sell order
                try:
                    alpaca.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side="sell",
                        type="market",
                        time_in_force=time_in_force
                    )
                    print(f"✅ Sold {quantity} of {symbol} at ${price:.2f} (Order Type: {time_in_force})")
                    log_trade(symbol, "SELL", quantity, price, reason)
                except Exception as e:
                    print(f"❌ Error executing trade for {symbol}: {e}")

            except Exception as e:
                print(f"❌ Error fetching position for {symbol}: {e}")

    except Exception as e:
        print(f"❌ Unexpected error in execute_trade(): {e}")


# ✅ Main Loop
def main():
    while True:
        for asset in ASSETS:
            print(f"📊 Fetching data for {asset}...")
            price = 0

            if asset in ["BTC-USD", "ETH-USD"]:
                price_data = fetch_crypto_data(asset)

                if price_data is None or price_data.empty:
                    print(f"❌ Failed to fetch price for {asset}")
                    continue  
    
                # ✅ Corrected `float()` conversion
                price = float(price_data["Close"].iloc[-1]) if isinstance(price_data["Close"], pd.Series) and not price_data.empty else 0.0
    
                print(f"💰 {asset} Price: ${price:.2f}")

            else:
                stock_data = fetch_asset_data(asset)
                if stock_data is None or stock_data.empty:
                    print(f"❌ No valid data for {asset}. Skipping...")
                    continue
                
                latest_data = stock_data.iloc[-1].to_dict()
                price = latest_data.get('Close', 0)

                if isinstance(price, pd.Series):
                    price = float(price.iloc[-1])
                elif price == 0 or price is None:
                    print(f"❌ Price data unavailable for {asset}")
                    continue

            # ✅ Get Decision and Extract Reason
            trade_decision = analyze_with_chatgpt(latest_data)
            print(f"📈 {asset} Decision: {trade_decision}")

            # ✅ Extract "BUY/SELL/HOLD" and the reason from the response
            if "DECISION:" in trade_decision and "REASON:" in trade_decision:
                decision_part = trade_decision.split("DECISION:")[1].strip()
                decision, reason = decision_part.split("REASON:", 1)
                decision = decision.strip()
                reason = reason.strip()
            else:
                decision = "HOLD"
                reason = "Could not extract reason from response."

            # ✅ Execute Trade
            execute_trade(asset, decision, price, reason)

        print("⏳ Waiting 5 minutes before next check...")
        time.sleep(300)

if __name__ == "__main__":
    main()
