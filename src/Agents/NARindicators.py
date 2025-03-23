import pandas as pd
import yfinance as yf


def get_stock_data(symbol, period="6mo"):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df if not df.empty else None

def calculate_rsi(df, period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculate the MACD indicator."""
    df["EMA12"] = df["Close"].ewm(span=short_window, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=long_window, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal Line"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_moving_averages(df, short_window=50, long_window=200):
    """Calculate short-term and long-term moving averages."""
    df["SMA50"] = df["Close"].rolling(window=short_window).mean()
    df["SMA200"] = df["Close"].rolling(window=long_window).mean()
    return df

def generate_technical_indicators(symbol):
    """Fetch stock data and compute indicators."""
    df = get_stock_data(symbol)
    if df is not None:
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_moving_averages(df)
        return df.iloc[-1][["RSI", "MACD", "Signal Line", "SMA50", "SMA200"]].to_dict()
    return None
