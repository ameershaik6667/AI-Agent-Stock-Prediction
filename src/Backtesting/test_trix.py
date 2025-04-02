# test_trix_investment.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from textwrap import dedent
import sys, os

# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -----------------------------------------------------
# Import Modules to Test from Our TRIX Codebase
# -----------------------------------------------------
# Import fetch_current_price and TrixInvestmentAdvisor from our TRIX UI module
from src.UI.trix_main import fetch_current_price, TrixInvestmentAdvisor
# Import calculate_trix from our TRIX indicator module
from src.Indicators.trix import calculate_trix
# Import DataFetcher for historical data tests
from src.Data_Retrieval.data_fetcher import DataFetcher

# Import Crew from crewai for integration testing
from crewai import Crew

# -----------------------------------------------------
# Dummy Classes for Monkeypatching (Simulating yfinance and yahooquery)
# -----------------------------------------------------
class DummyTicker:
    """
    A dummy Ticker class to simulate yfinance.Ticker and yahooquery.Ticker responses.
    """
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period=None, interval=None, start=None, end=None):
        # Create a dummy DataFrame with required columns.
        # For real-time tests, simulate 5 rows with timestamps (using minute frequency).
        dates = pd.date_range(start=datetime.today().date(), periods=5, freq='T')
        df = pd.DataFrame({
            'High': [100, 102, 101, 103, 104],
            'Low': [95, 96, 97, 98, 99],
            'Close': [98, 100, 99, 101, 102]
        }, index=dates)
        return df

    @property
    def info(self):
        # Return a dummy info dict for current stock price from yfinance.
        return {"regularMarketPrice": 150.0}

    @property
    def price(self):
        # Dummy implementation for yahooquery.Ticker price
        return {self.symbol: {"regularMarketPrice": 150.0}}

class DummyTickerMissingColumns:
    """
    A dummy Ticker class that returns a DataFrame missing the 'Close' column.
    Used to test error handling in calculate_trix.
    """
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period=None, interval=None, start=None, end=None):
        dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
        # 'Close' column is intentionally missing.
        df = pd.DataFrame({
            'High': [100, 102, 101, 103, 104],
            'Low': [95, 96, 97, 98, 99]
        }, index=dates)
        return df

    @property
    def info(self):
        return {"regularMarketPrice": 150.0}

    @property
    def price(self):
        return {self.symbol: {"regularMarketPrice": 150.0}}

# -----------------------------------------------------
# Unit Tests for fetch_current_price
# -----------------------------------------------------
def test_fetch_current_price(monkeypatch):
    """
    Test that fetch_current_price returns the correct current price using DummyTicker.
    """
    monkeypatch.setattr("src.UI.trix_main.Ticker", lambda symbol: DummyTicker(symbol))
    price = fetch_current_price("TEST")
    assert price == 150.0, "Expected current price to be 150.0 from DummyTicker"

def test_fetch_current_price_failure(monkeypatch):
    """
    Test that fetch_current_price handles exceptions and returns None.
    """
    def dummy_ticker(symbol):
        raise Exception("Test error")
    monkeypatch.setattr("src.UI.trix_main.Ticker", dummy_ticker)
    price = fetch_current_price("TEST")
    assert price is None, "Expected None when an exception occurs in fetch_current_price"

# -----------------------------------------------------
# Unit Tests for calculate_trix
# -----------------------------------------------------
def test_calculate_trix():
    """
    Test calculate_trix on a simple dataset with linearly increasing close prices.
    """
    # Create a DataFrame with 15 points of linearly increasing 'Close' values.
    df = pd.DataFrame({"Close": np.linspace(100, 110, 15)})
    df.index = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
    result = calculate_trix(df, length=3, signal=2)
    # Ensure TRIX and TRIX_SIGNAL columns are present.
    assert "TRIX" in result.columns, "Missing TRIX column"
    assert "TRIX_SIGNAL" in result.columns, "Missing TRIX_SIGNAL column"
    # Ensure that after the warm-up period, TRIX values are not NaN.
    assert not pd.isna(result["TRIX"].iloc[-1]), "Last TRIX value should not be NaN"

def test_calculate_trix_with_smoothing():
    """
    Test that additional smoothing produces a TRIX_SMOOTHED column.
    """
    df = pd.DataFrame({"Close": np.linspace(100, 110, 15)})
    df.index = pd.date_range(start="2023-01-01", periods=len(df), freq="D")
    result = calculate_trix(df, length=3, signal=2)
    result['TRIX_SMOOTHED'] = result['TRIX'].ewm(span=5, adjust=False).mean()
    assert "TRIX_SMOOTHED" in result.columns, "Missing TRIX_SMOOTHED column"
    assert not result['TRIX_SMOOTHED'].isna().all(), "TRIX_SMOOTHED values are all NaN"

def test_calculate_trix_missing_close():
    """
    Test that calculate_trix raises a KeyError when the 'Close' column is missing.
    """
    df = pd.DataFrame({"Open": [100, 101, 102]})
    df.index = pd.date_range(start="2023-01-01", periods=3, freq="D")
    with pytest.raises(KeyError):
        calculate_trix(df, length=3, signal=2)

# -----------------------------------------------------
# Unit Tests for TrixInvestmentAdvisor Integration
# -----------------------------------------------------
def test_trix_investment_advisor(monkeypatch):
    """
    Test that TrixInvestmentAdvisor generates a task containing the correct TRIX values and current price.
    """
    # Create a dummy TRIX DataFrame.
    data = {
        "TRIX": [0.5, 0.6, 0.7, 0.8, 0.9],
        "TRIX_SIGNAL": [0.55, 0.65, 0.75, 0.85, 0.95]
    }
    df_trix = pd.DataFrame(data)
    df_trix.index = pd.date_range(start="2023-01-01", periods=len(df_trix), freq="D")
    current_price = 150.0
    advisor = TrixInvestmentAdvisor()
    agent = advisor.advisor_agent()
    task = advisor.analyze_trix_and_price(agent, df_trix, current_price)
    # Verify the description includes the latest TRIX, TRIX_SIGNAL, and current price.
    assert "0.9" in task.description, "Latest TRIX value missing in task description"
    assert "0.95" in task.description, "Latest TRIX_SIGNAL value missing in task description"
    assert "150.0" in task.description, "Current stock price missing in task description"

def test_trix_investment_advisor_without_trix_signal():
    """
    Test TrixInvestmentAdvisor when the TRIX_SIGNAL column is missing.
    """
    data = {"TRIX": [0.7, 0.8, 0.9]}
    df_trix = pd.DataFrame(data)
    df_trix.index = pd.date_range(start="2023-01-01", periods=len(df_trix), freq="D")
    current_price = 155.0
    advisor = TrixInvestmentAdvisor()
    agent = advisor.advisor_agent()
    task = advisor.analyze_trix_and_price(agent, df_trix, current_price)
    # Verify that the description includes the latest TRIX and current price.
    assert "0.9" in task.description, "Latest TRIX value missing"
    assert "155.0" in task.description, "Current price missing"

# -----------------------------------------------------
# Integration Tests for Full Workflow
# -----------------------------------------------------
def test_full_workflow_historical(monkeypatch):
    """
    Test the complete workflow for historical data:
    - Fetch dummy historical data.
    - Calculate TRIX values.
    - Generate an investment decision using CrewAI.
    """
    # Monkeypatch DataFetcher.get_stock_data to return dummy historical data.
    def dummy_get_stock_data(symbol, start_date=None, end_date=None):
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        return pd.DataFrame({
            "High": [100, 102, 101, 103, 104],
            "Low": [95, 96, 97, 98, 99],
            "Close": [98, 100, 99, 101, 102]
        }, index=dates)
    monkeypatch.setattr(DataFetcher, "get_stock_data", dummy_get_stock_data)
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 5)
    data_fetcher = DataFetcher()
    data = data_fetcher.get_stock_data("TEST", start_date=start_date, end_date=end_date)
    assert data is not None

    # Calculate TRIX on the dummy data.
    trix_data = calculate_trix(data.copy(), length=3, signal=2)
    # Store dummy TRIX data in session-like dict.
    session_state = {"trix_data": trix_data}
    
    # Monkeypatch Ticker to use DummyTicker for current price.
    monkeypatch.setattr("src.UI.trix_main.Ticker", lambda symbol: DummyTicker(symbol))
    current_price = fetch_current_price("TEST")
    assert current_price == 150.0
    
    # Generate a CrewAI task using TrixInvestmentAdvisor.
    advisor = TrixInvestmentAdvisor()
    agent = advisor.advisor_agent()
    task = advisor.analyze_trix_and_price(agent, trix_data, current_price)
    
    # Monkeypatch Crew.kickoff to simulate a CrewAI response.
    class DummyCrew:
        def __init__(self, agents, tasks, verbose):
            self.agents = agents
            self.tasks = tasks
            self.verbose = verbose
        def kickoff(self):
            return "BUY recommendation based on TRIX data."
    monkeypatch.setattr("crewai.Crew", DummyCrew)
    from crewai import Crew  # Import after patching.
    
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert any(keyword in result for keyword in ["BUY", "SELL", "HOLD"]), "Result should contain a valid recommendation."

def test_full_workflow_realtime(monkeypatch):
    """
    Test the complete workflow for real-time data:
    - Fetch dummy real-time data using DummyTicker.
    - Calculate TRIX values.
    - Generate an investment decision using CrewAI.
    """
    # Monkeypatch Ticker.history to simulate real-time data.
    monkeypatch.setattr("src.UI.trix_main.Ticker", lambda symbol: DummyTicker(symbol))
    ticker = DummyTicker("TEST")
    data = ticker.history(period="1d", interval="1m")
    # Filter data to include only today's data.
    today_date = datetime.today().date()
    data = data[data.index.date == today_date]
    assert not data.empty, "Real-time data should not be empty"
    
    # Calculate TRIX on the dummy real-time data.
    trix_data = calculate_trix(data.copy(), length=3, signal=2)
    current_price = fetch_current_price("TEST")
    assert current_price == 150.0
    
    # Generate a CrewAI task using TrixInvestmentAdvisor.
    advisor = TrixInvestmentAdvisor()
    agent = advisor.advisor_agent()
    task = advisor.analyze_trix_and_price(agent, trix_data, current_price)
    
    # Monkeypatch Crew.kickoff to simulate a response.
    class DummyCrew:
        def __init__(self, agents, tasks, verbose):
            self.agents = agents
            self.tasks = tasks
            self.verbose = verbose
        def kickoff(self):
            return "HOLD recommendation based on TRIX data."
    monkeypatch.setattr("crewai.Crew", DummyCrew)
    from crewai import Crew
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert any(keyword in result for keyword in ["BUY", "SELL", "HOLD"]), "Result should contain a valid recommendation."

def test_fetch_stock_data_missing_columns(monkeypatch):
    """
    Test that calculate_trix fails when the fetched data is missing required columns.
    """
    # Monkeypatch DataFetcher.get_stock_data to return data missing the 'Close' column.
    def dummy_get_stock_data(symbol, start_date=None, end_date=None):
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        return pd.DataFrame({
            "High": [100, 102, 101, 103, 104],
            "Low": [95, 96, 97, 98, 99]
            # 'Close' column is missing
        }, index=dates)
    monkeypatch.setattr(DataFetcher, "get_stock_data", dummy_get_stock_data)
    data = DataFetcher().get_stock_data("TEST", start_date=datetime(2023, 1, 1), end_date=datetime(2023, 1, 5))
    with pytest.raises(KeyError):
        calculate_trix(data, length=3, signal=2)

def test_crewai_kickoff(monkeypatch):
    """
    Test that the CrewAI Crew kickoff returns a valid recommendation using a dummy kickoff.
    """
    # Monkeypatch Crew.kickoff to return a dummy result.
    def dummy_kickoff(self):
        return "SELL with strong conviction."
    monkeypatch.setattr(Crew, "kickoff", dummy_kickoff)
    
    # Create dummy data for TRIX calculation.
    dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
    data = pd.DataFrame({
        "High": [100, 102, 101],
        "Low": [95, 96, 97],
        "Close": [98, 100, 99]
    }, index=dates)
    trix_data = calculate_trix(data, length=3, signal=2)
    current_price = 160.0
    advisor = TrixInvestmentAdvisor()
    agent = advisor.advisor_agent()
    task = advisor.analyze_trix_and_price(agent, trix_data, current_price)
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == "SELL with strong conviction."

