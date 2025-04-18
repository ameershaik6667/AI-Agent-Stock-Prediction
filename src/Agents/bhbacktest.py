# backtest.py

import numpy as np
import pandas as pd


def calculate_macd(data, short=12, long=26, signal=9):
    data['EMA_short'] = data['Close'].ewm(span=short, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    return data

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def run_backtest(data):
    try:
        df = data.copy()
        df = calculate_macd(df)
        df = calculate_rsi(df)

        df.dropna(inplace=True)
        position = 0  # 0 means not holding, 1 means holding
        buy_price = 0
        initial_capital = 100000
        cash = initial_capital
        shares = 0

        for index, row in df.iterrows():
            macd = float(row['MACD'])
            signal = float(row['Signal'])
            rsi = float(row['RSI'])
            close = float(row['Close'])

            # Entry Signal
            if macd > signal and rsi < 70 and position == 0:
                buy_price = close
                shares = cash // buy_price
                cash -= shares * buy_price
                position = 1

            # Exit Signal
            elif (macd < signal or rsi > 70) and position == 1:
                sell_price = close
                cash += shares * sell_price
                position = 0
                shares = 0


        # At the end, sell if still holding
        if position == 1:
            cash += shares * df['Close'].iloc[-1]

        final_value = float(cash)  # Ensure native float

        return {
            "initial_capital": float(initial_capital),
            "final_portfolio_value": final_value,
            "return_percentage": round(((final_value - initial_capital) / initial_capital) * 100, 2)
        }


    except Exception as e:
        return {
            "initial_capital": 100000,
            "final_portfolio_value": 100000,
            "return_percentage": 0,
            "error": str(e)
        }
