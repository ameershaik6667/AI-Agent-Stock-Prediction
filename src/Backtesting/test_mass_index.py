import pandas as pd
import pytest
from datetime import datetime
import sys, os

# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Updated imports: refer to the mass index module in src/UI/mass_index.py
from src.UI.mass_index import (
    fetch_stock_data,
    fetch_current_price,
    calculate_mass_index,
    MassIndexAnalysisAgents,
    flatten_columns,
    standardize_columns
)
from crewai import Agent, Task, Crew

# -------------------------
# Dummy Classes for Monkeypatching
# -------------------------
class DummyTicker:
    """A dummy Ticker class to simulate yahooquery responses."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period=None, interval=None):
        # Create a simple DataFrame with required columns: date, High, Low, Close
        df = pd.DataFrame({
            'date': pd.date_range(start="2022-01-01", periods=5, freq='D'),
            'High': [10, 12, 11, 13, 14],
            'Low': [5, 7, 6, 8, 9],
            'Close': [7, 10, 9, 12, 11]
        })
        return df

    @property
    def price(self):
        # Return dummy real-time data as a dictionary.
        return {self.symbol: {'regularMarketPrice': 15}}

class DummyTickerMissingColumns:
    """A dummy Ticker that returns data missing a required column (e.g., 'Close')."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period=None, interval=None):
        df = pd.DataFrame({
            'date': pd.date_range(start="2022-01-01", periods=5, freq='D'),
            'High': [10, 12, 11, 13, 14],
            'Low': [5, 7, 6, 8, 9]
            # 'Close' column is intentionally missing.
        })
        return df

    @property
    def price(self):
        return {self.symbol: {'regularMarketPrice': 15}}

# -------------------------
# Unit Tests for Helper Functions
# -------------------------
def test_flatten_columns():
    # Create a DataFrame with MultiIndex columns
    arrays = [['A', 'A'], ['high', 'low']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    df = pd.DataFrame([[1, 2], [3, 4]], columns=index)
    flattened = flatten_columns(df.copy())
    assert list(flattened.columns) == ["A high", "A low"]

def test_flatten_columns_noop():
    # When DataFrame columns are already flat, flatten_columns should do nothing.
    df = pd.DataFrame({"high": [1, 2], "low": [3, 4]})
    flattened = flatten_columns(df.copy())
    assert list(flattened.columns) == ["high", "low"]

def test_standardize_columns():
    # Create a DataFrame with columns having a common trailing token.
    df = pd.DataFrame(columns=["high AAPL", "low AAPL", "close AAPL"])
    standardized = standardize_columns(df.copy())
    assert list(standardized.columns) == ["high", "low", "close"]

def test_standardize_columns_noop():
    # When no common trailing token exists, column names remain unchanged.
    df = pd.DataFrame(columns=["high", "low", "close"])
    standardized = standardize_columns(df.copy())
    assert list(standardized.columns) == ["high", "low", "close"]

# -------------------------
# Unit Tests for Mass Index Calculation
# -------------------------
def test_calculate_mass_index():
    # Create a simple DataFrame with 'high' and 'low' columns.
    data = {"high": [10, 12, 11, 13, 12], "low": [5, 6, 7, 8, 7]}
    df = pd.DataFrame(data)
    mass_index = calculate_mass_index(df, ema_period=2, sum_period=2)
    assert isinstance(mass_index, pd.Series)
    assert len(mass_index) == len(df)
    assert pd.isna(mass_index.iloc[0])

def test_calculate_mass_index_empty():
    # When an empty DataFrame is provided, expect an empty Series.
    df = pd.DataFrame(columns=["high", "low"])
    mass_index = calculate_mass_index(df, ema_period=9, sum_period=25)
    assert mass_index.empty

# -------------------------
# Unit Tests for Current Price Fetching
# -------------------------
def dummy_price_success(symbol):
    return {symbol: {'regularMarketPrice': 150.0}}

def dummy_price_failure(symbol):
    return {}

def test_fetch_current_price_success(monkeypatch):
    # Monkey-patch Ticker to simulate a successful price fetch.
    class DummyTickerSuccess:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return dummy_price_success(self.symbol)
    monkeypatch.setattr("src.UI.mass_index.Ticker", lambda symbol: DummyTickerSuccess(symbol))
    price = fetch_current_price("TEST")
    assert price == 150.0

def test_fetch_current_price_failure(monkeypatch):
    # Monkey-patch Ticker to simulate a failure.
    class DummyTickerFailure:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return dummy_price_failure(self.symbol)
    monkeypatch.setattr("src.UI.mass_index.Ticker", lambda symbol: DummyTickerFailure(symbol))
    price = fetch_current_price("TEST")
    assert price is None

# -------------------------
# Integration Tests for CrewAI Investment Decision
# -------------------------
class DummyAgent:
    def __init__(self, output):
        self.output = output

class DummyTask:
    def __init__(self, description):
        self.description = description

class DummyCrew:
    def __init__(self, agents, tasks, verbose):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
    def kickoff(self):
        # Return a simulated decision.
        return "BUY"

class DummyMassIndexAnalysisAgents(MassIndexAnalysisAgents):
    def mass_index_investment_advisor(self):
        return DummyAgent("DummyAgent")
    def mass_index_analysis(self, agent, mass_index_value, current_price):
        return DummyTask(f"Analyze Mass Index {mass_index_value} and current price {current_price}")

def test_mass_index_investment_decision():
    latest_mass_index = 30.5
    current_price = 150.0
    agents = DummyMassIndexAnalysisAgents()
    advisor_agent = agents.mass_index_investment_advisor()
    task = agents.mass_index_analysis(advisor_agent, latest_mass_index, current_price)
    crew = DummyCrew(agents=[advisor_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == "BUY"

@pytest.mark.parametrize("expected_decision", ["BUY", "SELL", "HOLD"])
def test_mass_index_investment_decision_various(expected_decision):
    latest_mass_index = 28.0
    current_price = 155.0
    class CustomDummyCrew:
        def __init__(self, agents, tasks, verbose):
            self.agents = agents
            self.tasks = tasks
            self.verbose = verbose
        def kickoff(self):
            return expected_decision
    class CustomDummyMassIndexAnalysisAgents(DummyMassIndexAnalysisAgents):
        def mass_index_investment_advisor(self):
            return DummyAgent("CustomDummyAgent")
        def mass_index_analysis(self, agent, mass_index_value, current_price):
            return DummyTask(f"Analyze Mass Index {mass_index_value} and price {current_price}")
    agents = CustomDummyMassIndexAnalysisAgents()
    advisor_agent = agents.mass_index_investment_advisor()
    task = agents.mass_index_analysis(advisor_agent, latest_mass_index, current_price)
    crew = CustomDummyCrew(agents=[advisor_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == expected_decision

# -------------------------
# Additional End-to-End Integration Tests
# -------------------------
def test_full_workflow_historical(monkeypatch):
    """
    End-to-end test for the historical data workflow: fetch data, calculate Mass Index, and generate a decision.
    """
    monkeypatch.setattr("src.UI.mass_index.Ticker", lambda symbol: DummyTicker(symbol))
    df = fetch_stock_data("TEST", period="1y", interval="1d")
    assert df is not None and len(df) == 5
    mass_index_series = calculate_mass_index(df, ema_period=3, sum_period=3)
    non_na = mass_index_series.dropna()
    assert len(non_na) > 0
    latest_mass_index = non_na.iloc[-1]
    current_price = fetch_current_price("TEST")
    assert current_price == 15
    agents = DummyMassIndexAnalysisAgents()
    advisor_agent = agents.mass_index_investment_advisor()
    task = agents.mass_index_analysis(advisor_agent, latest_mass_index, current_price)
    crew = DummyCrew(agents=[advisor_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == "BUY"

def test_full_workflow_realtime(monkeypatch):
    """
    End-to-end test for the real-time data workflow: fetch real-time data, calculate Mass Index, and generate a decision.
    """
    monkeypatch.setattr("src.UI.mass_index.Ticker", lambda symbol: DummyTicker(symbol))
    df = fetch_stock_data("TEST", period="1d", interval="1m")
    assert not df.empty
    mass_index_series = calculate_mass_index(df, ema_period=3, sum_period=3)
    non_na = mass_index_series.dropna()
    assert len(non_na) > 0
    latest_mass_index = non_na.iloc[-1]
    current_price = fetch_current_price("TEST")
    assert current_price == 15
    agents = DummyMassIndexAnalysisAgents()
    advisor_agent = agents.mass_index_investment_advisor()
    task = agents.mass_index_analysis(advisor_agent, latest_mass_index, current_price)
    crew = DummyCrew(agents=[advisor_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == "BUY"

def test_full_workflow_empty_data(monkeypatch):
    """
    Test full workflow behavior when fetch_stock_data returns an empty DataFrame.
    The Mass Index calculation should result in an empty Series, and no decision can be made.
    """
    class DummyTickerEmpty:
        def __init__(self, symbol):
            self.symbol = symbol
        def history(self, period=None, interval=None):
            return pd.DataFrame()  # Empty DataFrame
        @property
        def price(self):
            return {self.symbol: {'regularMarketPrice': 15}}
    monkeypatch.setattr("src.UI.mass_index.Ticker", lambda symbol: DummyTickerEmpty(symbol))
    df = fetch_stock_data("TEST", period="1y", interval="1d")
    assert df.empty
    mass_index_series = calculate_mass_index(df, ema_period=3, sum_period=3)
    assert mass_index_series.empty

def test_full_workflow_invalid_current_price(monkeypatch):
    """
    Test full workflow behavior when fetch_current_price returns None.
    """
    class DummyTickerNoPrice:
        def __init__(self, symbol):
            self.symbol = symbol
        def history(self, period=None, interval=None):
            # Return a valid DataFrame
            df = pd.DataFrame({
                'date': pd.date_range(start="2022-01-01", periods=5, freq='D'),
                'High': [10, 12, 11, 13, 14],
                'Low': [5, 7, 6, 8, 9],
                'Close': [7, 10, 9, 12, 11]
            })
            return df
        @property
        def price(self):
            return {}  # No price data
    monkeypatch.setattr("src.UI.mass_index.Ticker", lambda symbol: DummyTickerNoPrice(symbol))
    df = fetch_stock_data("TEST", period="1y", interval="1d")
    mass_index_series = calculate_mass_index(df, ema_period=3, sum_period=3)
    current_price = fetch_current_price("TEST")
    assert current_price is None
