import pandas as pd
import pytest
from datetime import datetime
import sys, os

# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Updated imports: refer to the main module in src/UI/donchian.py (which now contains the Donchian code)
from src.UI.donchian import (
    DonchianCalculator,
    fetch_stock_data,
    fetch_realtime_data,
    fetch_current_price,
    DonchianAnalysisAgents
)
from crewai import Agent, Task, Crew

# -------------------------
# Dummy Classes for Monkeypatching
# -------------------------
class DummyTicker:
    """A dummy Ticker class to simulate yahooquery responses."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None, period=None):
        # Create a simple DataFrame with required columns.
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
    """A dummy Ticker that returns data missing a required column (e.g. 'Close')."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None, period=None):
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
# Unit Tests for Data Fetching Functions
# -------------------------
def test_fetch_stock_data(monkeypatch):
    """Test historical data fetching with start and end dates."""
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTicker(symbol))
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 5)
    df = fetch_stock_data("TEST", start_date=start_date, end_date=end_date)
    assert df is not None
    for col in ['date', 'high', 'low', 'close']:
        assert col in df.columns
    assert len(df) == 5

def test_fetch_stock_data_no_dates(monkeypatch):
    """Test fetching historical data with fallback period when no dates are provided."""
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTicker(symbol))
    df = fetch_stock_data("TEST")
    assert df is not None
    for col in ['date', 'high', 'low', 'close']:
        assert col in df.columns

def test_fetch_realtime_data(monkeypatch):
    """Test real-time data fetching."""
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTicker(symbol))
    df_rt = fetch_realtime_data("TEST")
    assert df_rt is not None
    # The dummy returns a dict wrapped in a DataFrame.
    cell = df_rt.iloc[0]["TEST"]
    assert isinstance(cell, dict)
    assert cell.get('regularMarketPrice') == 15

def test_fetch_current_price(monkeypatch):
    """Test fetching the current stock price."""
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTicker(symbol))
    price = fetch_current_price("TEST")
    assert price == 15

# -------------------------
# Unit Tests for Donchian Calculator
# -------------------------
def test_donchian_calculator():
    """
    Test that the DonchianCalculator correctly computes the indicator:
      - 'donchian_high' as the highest high over the specified window.
      - 'donchian_low' as the lowest low over the specified window.
    """
    data = pd.DataFrame({
        'date': pd.date_range(start="2022-01-01", periods=5, freq='D'),
        'high': [10, 12, 11, 15, 14],
        'low': [5, 7, 6, 8, 9],
        'close': [7, 10, 9, 12, 11]
    })
    window = 3
    calc = DonchianCalculator(data, window=window)
    result = calc.calculate()
    # For the third row (index 2), the highest high over rows 0-2 is max([10,12,11]) = 12
    # and the lowest low is min([5,7,6]) = 5.
    assert result.iloc[2]['donchian_high'] == 12
    assert result.iloc[2]['donchian_low'] == 5

# -------------------------
# Unit Tests for CrewAI Analysis Agent
# -------------------------
def test_donchian_analysis():
    """
    Test that the DonchianAnalysisAgents generate a task with a description containing
    the expected indicator values and the current stock price.
    """
    data = pd.DataFrame({
        'donchian_high': [12, 13, 14],
        'donchian_low': [8, 7, 6]
    })
    current_price = 10
    agents = DonchianAnalysisAgents()
    advisor_agent = agents.donchian_investment_advisor()
    task = agents.donchian_analysis(advisor_agent, data, current_price)
    desc = task.description
    assert "12" in desc
    assert "6" in desc
    assert "10" in desc

# -------------------------
# Additional Integration Tests
# -------------------------
def test_full_workflow(monkeypatch):
    """
    Test the complete workflow with historical data:
    fetching stock data, calculating Donchian Channels, and generating an investment decision.
    """
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTicker(symbol))
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 5)
    data = fetch_stock_data("TEST", start_date=start_date, end_date=end_date)
    assert data is not None
    window = 3
    calc = DonchianCalculator(data, window=window)
    data_with_channels = calc.calculate()
    assert 'donchian_high' in data_with_channels.columns
    current_price = fetch_current_price("TEST")
    assert current_price == 15
    agents = DonchianAnalysisAgents()
    advisor_agent = agents.donchian_investment_advisor()
    task = agents.donchian_analysis(advisor_agent, data_with_channels, current_price)
    # Verify that the task description includes key phrases.
    assert "Donchian High" in task.description or "highest" in task.description
    assert "Current Stock Price" in task.description

def test_full_workflow_realtime(monkeypatch):
    """
    Test the complete workflow for real-time data:
    fetching real-time data, calculating the indicator (with window=1), and generating the analysis task.
    """
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTicker(symbol))
    data = fetch_realtime_data("TEST")
    assert data is not None
    # For real-time data, which is typically a single row, use a window of 1.
    calc = DonchianCalculator(data, window=1)
    data_with_channels = calc.calculate()
    assert not data_with_channels.empty
    current_price = fetch_current_price("TEST")
    assert current_price == 15
    agents = DonchianAnalysisAgents()
    advisor_agent = agents.donchian_investment_advisor()
    task = agents.donchian_analysis(advisor_agent, data_with_channels, current_price)
    assert "Donchian High" in task.description or "highest" in task.description

def test_fetch_stock_data_missing_columns(monkeypatch):
    """
    Test that fetch_stock_data returns None when required columns are missing.
    """
    monkeypatch.setattr("src.UI.donchian.Ticker", lambda symbol: DummyTickerMissingColumns(symbol))
    df = fetch_stock_data("TEST", start_date=datetime(2022, 1, 1), end_date=datetime(2022, 1, 5))
    assert df is None

def test_crewai_kickoff(monkeypatch):
    """
    Test that the CrewAI Crew kickoff returns a valid result using a dummy kickoff.
    """
    # Monkey-patch Crew.kickoff to return a dummy result.
    def dummy_kickoff(self):
        return "BUY with strong conviction."
    monkeypatch.setattr(Crew, "kickoff", dummy_kickoff)
    
    data = pd.DataFrame({
        'date': pd.date_range(start="2022-01-01", periods=3, freq='D'),
        'high': [10, 11, 12],
        'low': [5, 6, 7],
        'close': [7, 8, 9]
    })
    # Use a window of 1 for this small dataset.
    calc = DonchianCalculator(data, window=1)
    data_with_channels = calc.calculate()
    current_price = 10
    agents = DonchianAnalysisAgents()
    advisor_agent = agents.donchian_investment_advisor()
    task = agents.donchian_analysis(advisor_agent, data_with_channels, current_price)
    crew = Crew(
        agents=[advisor_agent],
        tasks=[task],
        verbose=True
    )
    result = crew.kickoff()
    assert result == "BUY with strong conviction."
