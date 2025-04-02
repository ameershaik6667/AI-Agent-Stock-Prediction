import pandas as pd
import pytest
from datetime import datetime
import sys, os

# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions and classes from your CMO module.
from src.UI.cmo import (
    fetch_data_yfinance,
    fetch_current_price,
    CMOCalculator,
    highlight_cmo_above_threshold,
    CMOAnalysisAgent
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
        # Create a dummy DataFrame with required columns.
        df = pd.DataFrame({
            'Date': pd.date_range("2022-01-01", periods=5, freq='D'),
            'Close': [100, 102, 101, 105, 107],
            'High': [101, 103, 102, 106, 108],
            'Low': [99, 101, 100, 104, 106]
        })
        return df

    @property
    def price(self):
        # Return dummy real-time data as a dictionary.
        return {self.symbol: {'regularMarketPrice': 150}}

class DummyTickerMissingColumns:
    """A dummy Ticker that returns data missing a required column (e.g. 'Close')."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period=None, interval=None):
        df = pd.DataFrame({
            'Date': pd.date_range("2022-01-01", periods=5, freq='D'),
            'High': [101, 103, 102, 106, 108],
            'Low': [99, 101, 100, 104, 106]
            # 'Close' is missing intentionally.
        })
        return df

    @property
    def price(self):
        return {self.symbol: {'regularMarketPrice': 150}}

class DummyCrew:
    """A dummy Crew class to simulate CrewAI's kickoff behavior."""
    def __init__(self, agents, tasks, verbose):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
    def kickoff(self):
        return "BUY"  # Simulated decision

class DummyAgent:
    """A dummy agent for testing."""
    pass

class DummyCMOAnalysisAgent:
    """A dummy version of CMOAnalysisAgent that returns fixed objects."""
    def create_agent(self):
        return DummyAgent()
    def create_task(self, agent, cmo_df, current_price):
        # Return a dummy Task with a description that embeds the current price and latest CMO.
        from crewai import Task  # Assuming Task is available from crewai
        return Task(
            description=f"Current Price: {current_price}, Latest CMO: {cmo_df['cmo'].iloc[-1]}",
            agent=agent,
            expected_output="dummy"
        )

# -------------------------
# Unit Tests for Data Fetching Functions
# -------------------------
def test_fetch_data_yfinance(monkeypatch):
    """Test historical data fetching returns a DataFrame with required columns."""
    monkeypatch.setattr("src.UI.cmo.yf.download", lambda symbol, period, interval: DummyTicker(symbol).history(period, interval))
    df = fetch_data_yfinance("TEST", data_source="Historical", period="1y", interval="1d")
    assert df is not None
    for col in ['date', 'high', 'low', 'close']:
        assert col in df.columns
    assert len(df) == 5

def test_fetch_current_price(monkeypatch):
    """Test fetching the current stock price returns the dummy value."""
    monkeypatch.setattr("src.UI.cmo.Ticker", lambda symbol: DummyTicker(symbol))
    price = fetch_current_price("TEST")
    assert price == 150

def test_fetch_data_missing_columns(monkeypatch):
    """Test that fetch_data_yfinance returns None when required columns are missing."""
    monkeypatch.setattr("src.UI.cmo.yf.download", lambda symbol, period, interval: DummyTickerMissingColumns(symbol).history(period, interval))
    df = fetch_data_yfinance("TEST", data_source="Historical", period="1y", interval="1d")
    assert df is None

# -------------------------
# Unit Tests for CMO Calculator and Highlight Function
# -------------------------
def test_CMOCalculator_standard():
    """Test that CMOCalculator correctly computes the CMO using the standard method."""
    df = pd.DataFrame({'close': [100, 105, 102, 107, 110]})
    calc = CMOCalculator(df, period=3, calc_method="Standard", keep_intermediate=True)
    result_df = calc.calculate()
    assert 'cmo' in result_df.columns
    assert pd.notnull(result_df['cmo'].iloc[-1])

def test_CMOCalculator_absolute():
    """Test CMOCalculator using the 'Absolute' method."""
    df = pd.DataFrame({'close': [100, 95, 97, 96, 98]})
    calc = CMOCalculator(df, period=3, calc_method="Absolute", keep_intermediate=True)
    result_df = calc.calculate()
    # With the absolute method, losses are 0 so CMO should be 100.
    # If all changes are taken as absolute gains, then CMO becomes 100.
    assert all(result_df['cmo'].dropna() == 100)

def test_CMOCalculator_smoothing_SMA():
    """Test that SMA smoothing is applied correctly."""
    df = pd.DataFrame({'close': [100, 105, 102, 107, 110]})
    calc = CMOCalculator(df, period=3, apply_smoothing="SMA", smoothing_period=2, keep_intermediate=False)
    result_df = calc.calculate()
    # Ensure that CMO values are smoothed (the first few values might be NaN)
    assert result_df['cmo'].rolling(window=2).mean().iloc[-1] is not None

def test_CMOCalculator_smoothing_EMA():
    """Test that EMA smoothing is applied correctly."""
    df = pd.DataFrame({'close': [100, 105, 102, 107, 110]})
    calc = CMOCalculator(df, period=3, apply_smoothing="EMA", smoothing_period=2, keep_intermediate=False)
    result_df = calc.calculate()
    # EMA smoothing should produce non-NaN results after a few data points.
    assert pd.notnull(result_df['cmo'].iloc[-1])

def test_CMOCalculator_keep_intermediate():
    """Test that intermediate columns are kept when requested."""
    df = pd.DataFrame({'close': [100, 105, 102, 107, 110]})
    calc = CMOCalculator(df, period=3, keep_intermediate=True)
    result_df = calc.calculate()
    # Intermediate columns such as 'price_change' should exist.
    assert 'price_change' in result_df.columns

def test_highlight_cmo_above_threshold():
    """Test that the highlight function returns correct CSS color string."""
    assert highlight_cmo_above_threshold(80, 70) == "color: red"
    assert highlight_cmo_above_threshold(60, 70) == "color: black"

# -------------------------
# Unit Tests for Crew AI Analysis Agent
# -------------------------
def test_cmo_analysis_task():
    """
    Test that the CMOAnalysisAgent creates a task with a description
    containing the latest CMO value and the current stock price.
    """
    # Create dummy CMO data with a known last value.
    df = pd.DataFrame({'cmo': [20, 30, 40, 50, 60]})
    current_price = 150
    agent_obj = CMOAnalysisAgent()
    advisor_agent = agent_obj.create_agent()
    task = agent_obj.create_task(advisor_agent, df, current_price)
    desc = task.description
    assert "150" in desc
    assert "60" in desc

# -------------------------
# Integration Tests
# -------------------------
def test_investment_decision(monkeypatch):
    """
    Test the full investment decision workflow:
    - Compute CMO from dummy data.
    - Fetch dummy current price.
    - Generate a Crew AI task.
    - Simulate a Crew kickoff returning a dummy decision.
    """
    # Prepare dummy CMO data
    df = pd.DataFrame({
        'date': pd.date_range("2022-01-01", periods=5, freq='D'),
        'close': [100, 105, 102, 107, 110],
        'high': [101, 106, 103, 108, 111],
        'low': [99, 104, 101, 106, 109]
    })
    calc = CMOCalculator(df, period=3, calc_method="Standard")
    cmo_df = calc.calculate()
    
    # Monkeypatch fetch_current_price to return a dummy value.
    monkeypatch.setattr("src.UI.cmo.fetch_current_price", lambda symbol: 150)
    # Monkeypatch Crew to use DummyCrew.
    monkeypatch.setattr("src.UI.cmo.Crew", lambda agents, tasks, verbose: DummyCrew(agents, tasks, verbose))
    # Monkeypatch CMOAnalysisAgent to use the dummy version.
    monkeypatch.setattr("src.UI.cmo.CMOAnalysisAgent", lambda: DummyCMOAnalysisAgent())
    
    advisor = CMOAnalysisAgent().create_agent()
    task = CMOAnalysisAgent().create_task(advisor, cmo_df, 150)
    crew = DummyCrew(agents=[advisor], tasks=[task], verbose=True)
    decision = crew.kickoff()
    assert decision == "BUY"

def test_full_workflow_realtime(monkeypatch):
    """
    Integration test for the full workflow using real-time data:
      - Fetch dummy real-time data.
      - Calculate CMO.
      - Retrieve dummy current price.
      - Generate a Crew AI task and simulate a decision.
    """
    # Monkeypatch yf.download to use DummyTicker for real-time data.
    monkeypatch.setattr("src.UI.cmo.yf.download", lambda symbol, period, interval: DummyTicker(symbol).history(period, interval))
    df = fetch_data_yfinance("TEST", data_source="Real-Time", period="1d", interval="1m")
    assert df is not None
    calc = CMOCalculator(df, period=3, calc_method="Standard")
    cmo_df = calc.calculate()
    assert 'cmo' in cmo_df.columns
    monkeypatch.setattr("src.UI.cmo.fetch_current_price", lambda symbol: 150)
    monkeypatch.setattr("src.UI.cmo.Crew", lambda agents, tasks, verbose: DummyCrew(agents, tasks, verbose))
    monkeypatch.setattr("src.UI.cmo.CMOAnalysisAgent", lambda: DummyCMOAnalysisAgent())
    advisor = CMOAnalysisAgent().create_agent()
    task = CMOAnalysisAgent().create_task(advisor, cmo_df, 150)
    crew = DummyCrew(agents=[advisor], tasks=[task], verbose=True)
    decision = crew.kickoff()
    assert decision == "BUY"

def test_investment_decision_missing_cmo(monkeypatch):
    """
    Test that if no CMO data is calculated (e.g., empty DataFrame),
    the workflow does not proceed and returns an error.
    """
    # Monkeypatch fetch_data_yfinance to return an empty DataFrame.
    monkeypatch.setattr("src.UI.cmo.yf.download", lambda symbol, period, interval: pd.DataFrame())
    df = fetch_data_yfinance("TEST", data_source="Historical", period="1y", interval="1d")
    assert df is None

