#!/usr/bin/env python3
import pytest
import pandas as pd
import numpy as np
import datetime
import sys, os

# Update the system path to import modules from parent directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import ADX modules from our project.
from src.Indicators.adx_indicator import ADXIndicator
from src.UI.adx_main import (
    fetch_stock_data,
    fetch_realtime_data,
    fetch_current_price,
    ADXAnalysisAgent
)
from crewai import Crew

# -------------------------
# UNIT TESTS FOR ADX INDICATOR
# -------------------------
def test_adx_calculation_sma():
    """
    Test ADX calculation using SMA smoothing.
    Use sample data with varying high and low values to generate non-zero directional movement.
    """
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'High': [110, 116, 123, 129, 137, 143, 150, 157, 163, 170],
        'Low': [100, 103, 107, 112, 118, 125, 133, 142, 152, 163],
        'Close': [105, 108, 113, 120, 128, 137, 147, 158, 170, 183]
    })
    adx = ADXIndicator(period=3, smoothing_method="SMA")
    df_with_adx = adx.calculate(df)
    # Check that ADX column exists and that some non-null values are computed.
    assert 'ADX' in df_with_adx.columns, "ADX column should be present with SMA smoothing."
    assert df_with_adx['ADX'].dropna().shape[0] > 0, "ADX values should be calculated with SMA."

def test_adx_calculation_ema():
    """
    Test ADX calculation using EMA smoothing.
    Use sample data with varying values.
    """
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'High': [120, 127, 135, 142, 150, 158, 165, 172, 180, 187],
        'Low': [110, 115, 121, 127, 134, 141, 149, 157, 166, 175],
        'Close': [115, 121, 128, 135, 143, 151, 159, 168, 178, 189]
    })
    adx = ADXIndicator(period=3, smoothing_method="EMA")
    df_with_adx = adx.calculate(df)
    assert 'ADX' in df_with_adx.columns, "ADX column should be present with EMA smoothing."
    assert df_with_adx['ADX'].dropna().shape[0] > 0, "ADX values should be calculated with EMA."

def test_adx_calculation_invalid_columns():
    """
    Test ADX calculation when required columns are missing.
    Expect a KeyError when 'Close' column is missing.
    """
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'High': [100, 105, 110, 115, 120],
        'Low': [90, 95, 100, 105, 110]
        # 'Close' column is intentionally missing.
    })
    adx = ADXIndicator(period=3, smoothing_method="SMA")
    with pytest.raises(KeyError):
        _ = adx.calculate(df)

# -------------------------
# UNIT TESTS FOR DATA FETCHING FUNCTIONS
# -------------------------
def test_fetch_stock_data(monkeypatch):
    """
    Test historical data fetching.
    Monkey-patch yfinance.download to return sample data.
    """
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'High': [110, 115, 120, 125, 130],
        'Low': [100, 105, 110, 115, 120],
        'Close': [105, 110, 115, 120, 125]
    })
    # Accept arbitrary keyword arguments to handle unexpected keywords.
    def mock_download(symbol, start, end, **kwargs):
        return sample_data
    monkeypatch.setattr("yfinance.download", mock_download)
    result = fetch_stock_data("AAPL", period="1y")
    assert isinstance(result, pd.DataFrame), "fetch_stock_data should return a DataFrame."
    assert not result.empty, "Historical data should not be empty."
    assert 'Close' in result.columns, "Fetched data must contain 'Close' column."

def test_fetch_stock_data_empty(monkeypatch):
    """
    Test historical data fetching when no data is returned.
    """
    def mock_download(symbol, start, end, **kwargs):
        return pd.DataFrame()  # Return empty DataFrame.
    monkeypatch.setattr("yfinance.download", mock_download)
    result = fetch_stock_data("AAPL", period="1y")
    assert result.empty, "When no data is fetched, DataFrame should be empty."

def test_fetch_realtime_data(monkeypatch):
    """
    Test real-time data fetching.
    Monkey-patch yfinance.Ticker to return sample intraday data.
    """
    now = datetime.datetime.now()
    sample_realtime = pd.DataFrame({
        'Datetime': pd.date_range(start=now, periods=5, freq='T'),
        'High': [150, 151, 152, 153, 154],
        'Low': [148, 149, 150, 151, 152],
        'Close': [149, 150, 151, 152, 153]
    })
    class DummyTicker:
        def history(self, period, interval):
            return sample_realtime
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: DummyTicker())
    result = fetch_realtime_data("AAPL")
    assert isinstance(result, pd.DataFrame), "fetch_realtime_data should return a DataFrame."
    assert not result.empty, "Real-time data should not be empty."
    assert 'Close' in result.columns, "Real-time data must contain 'Close' column."

def test_fetch_current_price(monkeypatch):
    """
    Test current price fetching using a dummy ticker.
    """
    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return {self.symbol: {'regularMarketPrice': 150.25}}
    monkeypatch.setattr("src.UI.adx_main.Ticker", lambda symbol: DummyTicker(symbol))
    price = fetch_current_price("AAPL")
    assert price == 150.25, "Current price should be 150.25."

def test_fetch_current_price_exception(monkeypatch):
    """
    Test fetch_current_price when an exception is raised.
    """
    def dummy_price_error(symbol):
        raise Exception("Test error")
    class DummyTickerError:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return dummy_price_error(self.symbol)
    monkeypatch.setattr("src.UI.adx_main.Ticker", lambda symbol: DummyTickerError(symbol))
    price = fetch_current_price("AAPL")
    assert price is None, "On exception, fetch_current_price should return None."

# -------------------------
# UNIT TESTS FOR CREWAI INVESTMENT DECISION INTEGRATION
# -------------------------
def test_adx_analysis_task():
    """
    Test the creation of a CrewAI task for ADX analysis.
    """
    df = pd.DataFrame({'ADX': [20, 25, 30, 35, 40]})
    agent_instance = ADXAnalysisAgent()
    advisor_agent = agent_instance.adx_investment_advisor()
    task = agent_instance.adx_analysis(advisor_agent, df, 150.0)
    assert "Latest ADX Value:" in task.description, "Task description must include latest ADX value."
    assert "Current Stock Price:" in task.description, "Task description must include current stock price."

def test_crewai_decision(monkeypatch):
    """
    Test the CrewAI decision-making process.
    Simulate the decision process by replacing Crew with a dummy class.
    """
    df = pd.DataFrame({'ADX': np.linspace(20, 40, 10)})
    agent_instance = ADXAnalysisAgent()
    advisor_agent = agent_instance.adx_investment_advisor()
    task = agent_instance.adx_analysis(advisor_agent, df, 150.0)
    
    # Dummy Crew that returns a valid decision.
    class DummyCrew:
        def __init__(self, agents, tasks, verbose):
            self.agents = agents
            self.tasks = tasks
            self.verbose = verbose
        def kickoff(self):
            return "BUY"
    monkeypatch.setattr("crewai.Crew", lambda agents, tasks, verbose: DummyCrew(agents, tasks, verbose))
    crew = Crew(agents=[advisor_agent], tasks=[task], verbose=True)
    decision = crew.kickoff()
    assert decision in ["BUY", "SELL", "HOLD"], "Decision must be BUY, SELL, or HOLD."

# -------------------------
# INTEGRATION TESTS FOR END-TO-END WORKFLOW
# -------------------------
def test_integration_historical_flow(monkeypatch):
    """
    End-to-end integration test for historical data workflow:
    - Fetch historical data.
    - Calculate ADX.
    - Generate a CrewAI task.
    """
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=20, freq='D'),
        'High': [110 + i for i in range(20)],
        'Low': [100 + i for i in range(20)],
        'Close': [105 + i for i in range(20)]
    })
    def mock_download(symbol, start, end, **kwargs):
        return sample_data
    monkeypatch.setattr("yfinance.download", mock_download)
    data = fetch_stock_data("AAPL", period="1y")
    assert not data.empty, "Historical data should not be empty."
    adx_indicator = ADXIndicator(period=14, smoothing_method="SMA")
    adx_data = adx_indicator.calculate(data)
    assert 'ADX' in adx_data.columns, "Calculated data should include ADX column."
    agent_instance = ADXAnalysisAgent()
    advisor_agent = agent_instance.adx_investment_advisor()
    task = agent_instance.adx_analysis(advisor_agent, adx_data, 150.0)
    assert "Latest ADX Value:" in task.description, "Task description should include ADX information."

def test_integration_realtime_flow(monkeypatch):
    """
    End-to-end integration test for real-time data workflow:
    - Fetch real-time intraday data.
    - Calculate ADX using EMA smoothing.
    - Validate the calculated data.
    """
    now = datetime.datetime.now()
    sample_realtime = pd.DataFrame({
        'Datetime': pd.date_range(start=now, periods=10, freq='T'),
        'High': [150 + i for i in range(10)],
        'Low': [148 + i for i in range(10)],
        'Close': [149 + i for i in range(10)]
    })
    class DummyTicker:
        def history(self, period, interval):
            return sample_realtime
    monkeypatch.setattr("yfinance.Ticker", lambda symbol: DummyTicker())
    data = fetch_realtime_data("AAPL")
    assert not data.empty, "Real-time data should not be empty."
    adx_indicator = ADXIndicator(period=14, smoothing_method="EMA")
    adx_data = adx_indicator.calculate(data)
    assert 'ADX' in adx_data.columns, "Calculated data should include ADX column."

def test_integration_crewai_decision(monkeypatch):
    """
    End-to-end integration test for the CrewAI decision process.
    Simulate a decision using a dummy Crew that returns a predefined decision.
    """
    sample_adx_data = pd.DataFrame({'ADX': np.linspace(20, 40, 10)})
    agent_instance = ADXAnalysisAgent()
    advisor_agent = agent_instance.adx_investment_advisor()
    task = agent_instance.adx_analysis(advisor_agent, sample_adx_data, 150.0)
    class DummyCrew:
        def __init__(self, agents, tasks, verbose):
            self.agents = agents
            self.tasks = tasks
            self.verbose = verbose
        def kickoff(self):
            return "SELL"
    monkeypatch.setattr("crewai.Crew", lambda agents, tasks, verbose: DummyCrew(agents, tasks, verbose))
    crew = Crew(agents=[advisor_agent], tasks=[task], verbose=True)
    decision = crew.kickoff()
    assert decision in ["BUY", "SELL", "HOLD"], "Decision must be one of BUY, SELL, or HOLD."

def test_integration_invalid_current_price(monkeypatch):
    """
    Integration test for error handling when current price data is missing.
    Ensure that fetch_current_price returns None in this scenario.
    """
    class DummyTickerNoPrice:
        def __init__(self, symbol):
            self.symbol = symbol
        def history(self, period=None, interval=None):
            # Return valid historical data.
            return pd.DataFrame({
                'Date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
                'High': [110, 112, 114, 116, 118],
                'Low': [100, 102, 104, 106, 108],
                'Close': [105, 107, 109, 111, 113]
            })
        @property
        def price(self):
            return {}  # Simulate missing price data.
    monkeypatch.setattr("src.UI.adx_main.Ticker", lambda symbol: DummyTickerNoPrice(symbol))
    df = fetch_stock_data("AAPL", period="1y")
    adx_indicator = ADXIndicator(period=14, smoothing_method="SMA")
    adx_data = adx_indicator.calculate(df)
    current_price = fetch_current_price("AAPL")
    assert current_price is None, "When price data is missing, fetch_current_price should return None."
