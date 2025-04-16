import json
import logging
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize OpenAI API Client
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_stock_symbol(query):
    """Extracts stock symbol and key financial indicators from the user query using OpenAI."""
    system_prompt = (
        "You are a financial AI specializing in parsing stock-related queries. "
        "Extract only the stock symbols and financial indicators mentioned in the query. "
        "Return data in strict JSON format. Example output: {\"symbol\": \"AAPL\", \"indicators\": [\"moving_average\", \"RSI\"]}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        
        # Ensure OpenAI response is properly formatted JSON
        content = response.choices[0].message.content.strip()
        # Remove triple backticks and 'json' label if present
        content = re.sub(r"```json\s*|\s*```", "", content).strip()
        
        # Check if response starts with '{' and ends with '}'
        if not (content.startswith("{") and content.endswith("}")):
            logging.error(f"Invalid JSON response: {content}")
            return None

        return json.loads(content)
    
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error extracting stock symbol: {e}")
        return None

def get_stock_data(symbol, period="6mo"):
    """Fetches stock data from Yahoo Finance and calculates indicators."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)

        if hist.empty:
            logging.warning("No stock data found.")
            return None

        hist["50_MA"] = hist["Close"].rolling(window=50).mean()
        hist["200_MA"] = hist["Close"].rolling(window=200).mean()
        hist["RSI"] = 100 - (100 / (1 + (hist["Close"].pct_change().mean() / hist["Close"].pct_change().std())))

        indicators = {
            "moving_average_50": hist["50_MA"].iloc[-1],
            "moving_average_200": hist["200_MA"].iloc[-1],
            "RSI": hist["RSI"].iloc[-1]
        }
        
        return {
    "symbol": symbol,
    "indicators": indicators,
    "history": hist.tail(10).to_dict(orient="records")
}


    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return None

def backtest_strategy(stock_data, strategy="moving_average"):
    """Backtests a simple strategy based on historical stock data or a textual recommendation."""
    try:
        df = pd.DataFrame(stock_data["history"])

        # Ensure 'Date' exists as a column
        if "Date" not in df.columns:
            df.index.name = "Date"
            df.reset_index(inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        df["Signal"] = 0  # Default = Hold

        # NEW: if strategy is a recommendation string, set signals based on it
        if isinstance(strategy, str) and strategy.lower() in ["buy", "sell", "hold"]:
            rec = strategy.lower()
            if rec == "buy":
                df["Signal"] = 1
            elif rec == "sell":
                df["Signal"] = -1
            else:
                df["Signal"] = 0

        # OLD STRATEGY LOGIC: moving average or RSI
        elif strategy == "moving_average":
            df["Signal"] = (
                (df["50_MA"] > df["200_MA"]) & (df["50_MA"].shift(1) <= df["200_MA"].shift(1))
            ).astype(int)
            df["Signal"] = df["Signal"].where(df["Signal"] == 1, -(
                (df["50_MA"] < df["200_MA"]) & (df["50_MA"].shift(1) >= df["200_MA"].shift(1))
            ).astype(int))

        elif strategy == "RSI":
            df["Signal"] = df["RSI"].apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0))

        # Continue with the rest of the logic
        df["Position"] = df["Signal"].replace(to_replace=0, method="ffill")  # Maintain position
        df["Daily Return"] = df["Close"].pct_change()
        df["Strategy Return"] = df["Position"].shift(1) * df["Daily Return"]

        total_return = df["Strategy Return"].sum()
        cumulative_return = (1 + df["Strategy Return"]).cumprod().iloc[-1]

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cumulative_return_pct": round((cumulative_return - 1) * 100, 2),
            "backtest_data": df.tail(10).to_dict(orient="records")
        }

    except Exception as e:
        logging.error(f"Error during backtesting: {e}")
        return None



def fetch_market_sentiment(symbol):
    """Fetches news sentiment data using the correct Serper API endpoint."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,  
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "q": f"{symbol} stock news"
    })

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching market sentiment data: {e}")
        return None

def analyze_market_data(stock_data, sentiment_data):
    """Uses OpenAI to analyze stock data and market sentiment, providing actionable recommendations."""

    system_prompt = (
        "You are a financial AI analyzing stock performance. Given stock indicators and news sentiment, "
        "predict whether it is a good time to buy, hold, or sell. "
        "Provide clear, data-backed investment recommendations."
        "\n\nFormat your response as follows:\n"
        "1️ **Stock Analysis**: Summary of key indicators.\n"
        "2️ **Market Sentiment**: Brief sentiment analysis.\n"
        "3️ **Actionable Recommendation**: Buy/Hold/Sell with reasoning."
    )

    user_prompt = f"Stock Data: {json.dumps(stock_data)}\nMarket Sentiment: {json.dumps(sentiment_data)}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        recommendation = response.choices[0].message.content
        return recommendation

    except Exception as e:
        logging.error(f"Error analyzing market data: {e}")
        return "Error: Could not generate recommendation."

def visualize_stock_data(symbol, stock_data):
    """Generates a visualization of stock trends, moving averages, and RSI indicators."""
    
    # Convert history back to DataFrame
    history_df = pd.DataFrame(stock_data["history"])

    # Ensure 'Date' column is set as index if not already
    if "Date" in history_df.columns:
        history_df["Date"] = pd.to_datetime(history_df["Date"])  # Convert to datetime
        history_df.set_index("Date", inplace=True)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Price Chart
    axs[0].plot(history_df.index, history_df["Close"], label="Closing Price", color="blue")
    axs[0].plot(history_df.index, history_df["50_MA"], label="50-Day MA", linestyle="--", color="orange")
    axs[0].plot(history_df.index, history_df["200_MA"], label="200-Day MA", linestyle="--", color="red")
    axs[0].set_title(f"{symbol} Stock Price & Moving Averages")
    axs[0].set_ylabel("Price (USD)")
    axs[0].legend()

    # RSI Chart
    axs[1].plot(history_df.index, history_df["RSI"], label="RSI", color="purple")
    axs[1].axhline(70, linestyle="--", color="red", label="Overbought (70)")
    axs[1].axhline(30, linestyle="--", color="green", label="Oversold (30)")
    axs[1].set_title(f"{symbol} RSI Indicator")
    axs[1].set_ylabel("RSI Value")
    axs[1].set_xlabel("Date")
    axs[1].legend()

    plt.tight_layout()
    plt.show()