import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

#  Fetch Real Tesla Stock Data
def fetch_real_data(symbol="TSLA", start_date="2023-01-01", end_date=None):
    end_date = end_date or datetime.today().strftime('%Y-%m-%d')
    df = yf.download(symbol, start=start_date, end=end_date)
    df.index = pd.to_datetime(df.index)
    
    # 🔹 Handle missing values
    return df['Close'].dropna()

#  Scenario Simulation Agent
class ScenarioSimulationAgent:
    @staticmethod
    def generate_market_scenario(dates, base_price=100, scenario_type="high_volatility"):
        np.random.seed(42)
        prices = [base_price]

        for _ in range(1, len(dates)):
            if scenario_type == "high_volatility":
                daily_return = np.random.normal(0, 5)
            elif scenario_type == "bullish_trend":
                daily_return = np.random.normal(0.5, 1)
            elif scenario_type == "bearish_trend":
                daily_return = np.random.normal(-0.5, 1)
            elif scenario_type == "sudden_spike":
                daily_return = np.random.normal(0, 1)
                if np.random.rand() > 0.95:
                    daily_return += np.random.choice([20, -20])
            else:
                daily_return = np.random.normal(0, 1)

            prices.append(prices[-1] + daily_return)

        return np.array(prices)

#  Indicator Calculations
def calculate_rsi(prices, period=14):
    prices = np.asarray(prices).flatten()  # Ensure 1D array
    
    if len(prices) == 0:
        raise ValueError("❌ Price data is empty. Check data fetching process.")

    delta = np.diff(prices, prepend=prices[0])  # Calculate price changes
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    prices = np.asarray(prices).flatten()  # Ensure 1D array
    
    if len(prices) == 0:
        raise ValueError("❌ Price data is empty. Check data fetching process.")

    short_ema = pd.Series(prices).ewm(span=short_window, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    return macd.values, signal.values

#  Visualization Agent
class VisualizationAgent:
    @staticmethod
    def plot_comparison(dates, real_prices, simulated_prices, real_rsi, simulated_rsi, real_macd, real_signal, simulated_macd, simulated_signal):
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Real TSLA Price", "Simulated High Volatility",
                            "RSI (Real)", "RSI (Simulated)",
                            "MACD (Real)", "MACD (Simulated)"),
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # Price Comparison
        fig.add_trace(go.Scatter(x=dates, y=real_prices, mode='lines', name='Real TSLA', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=simulated_prices, mode='lines', name='Simulated', line=dict(color='orange')), row=1, col=2)

        #  RSI Comparison
        fig.add_trace(go.Scatter(x=dates, y=real_rsi, mode='lines', name='RSI (Real)', line=dict(color='purple')), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=simulated_rsi, mode='lines', name='RSI (Simulated)', line=dict(color='red')), row=2, col=2)

        #  MACD Comparison
        fig.add_trace(go.Scatter(x=dates, y=real_macd, mode='lines', name='MACD (Real)', line=dict(color='green')), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=real_signal, mode='lines', name='Signal (Real)', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=simulated_macd, mode='lines', name='MACD (Simulated)', line=dict(color='green', dash='dot')), row=3, col=2)
        fig.add_trace(go.Scatter(x=dates, y=simulated_signal, mode='lines', name='Signal (Simulated)', line=dict(color='red', dash='dot')), row=3, col=2)

        fig.update_layout(title="Real Tesla Data vs Simulated High Volatility", template="plotly_white", height=900, width=1200)
        fig.show()

#  Scenario Testing
if __name__ == "__main__":
    #  Fetch Real Tesla Data
    real_prices = fetch_real_data(symbol="TSLA", start_date="2023-01-01")
    dates = real_prices.index

    #  Simulate High Volatility Scenario
    simulated_prices = ScenarioSimulationAgent.generate_market_scenario(dates, base_price=real_prices.iloc[0], scenario_type="high_volatility")

    #  Calculate Indicators
    real_rsi = calculate_rsi(real_prices.values)
    simulated_rsi = calculate_rsi(simulated_prices)
    real_macd, real_signal = calculate_macd(real_prices.values)
    simulated_macd, simulated_signal = calculate_macd(simulated_prices)

    #  Compare Real vs Simulated Data
    VisualizationAgent.plot_comparison(dates, real_prices, simulated_prices, real_rsi, simulated_rsi, real_macd, real_signal, simulated_macd, simulated_signal)
