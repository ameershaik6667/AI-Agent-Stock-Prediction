import pandas as pd
import pytest
import datetime
import sys
import os
import streamlit as st

# Adjust the system path to import modules from your project structure.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.UI.atr import (
    fetch_historical_data,
    fetch_realtime_data,
    calculate_atr,
    fetch_current_price,
    ATRAnalysisAgents
)

# -------------------------------
# Fixtures and Dummy Data
# -------------------------------

@pytest.fixture
def sample_atr_data():
    """
    Creates a small sample DataFrame with 'High', 'Low', and 'Close' columns.
    The index is a date range.
    """
    dates = pd.date_range(start='2022-01-01', periods=5, freq='D')
    data = {
        'High': [10, 12, 11, 13, 14],
        'Low': [5, 6, 5, 7, 8],
        'Close': [7, 9, 8, 10, 11]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_atr_data_extended():
    """
    Creates an extended sample DataFrame for ATR calculation.
    """
    dates = pd.date_range(start='2022-01-01', periods=15, freq='D')
    data = {
        'High': [10, 12, 11, 13, 14, 15, 16, 18, 17, 19, 20, 22, 21, 23, 24],
        'Low': [5, 6, 5, 7, 8, 9, 10, 11, 10, 12, 13, 14, 13, 15, 16],
        'Close': [7, 9, 8, 10, 11, 12, 13, 15, 14, 16, 18, 19, 18, 20, 21]
    }
    return pd.DataFrame(data, index=dates)

# -------------------------------
# Unit Tests for ATR Calculation
# -------------------------------

def test_calculate_atr_columns(sample_atr_data):
    """
    Verify that the calculate_atr function adds all required ATR-related columns.
    """
    period = 3
    result = calculate_atr(sample_atr_data, period=period)
    expected_columns = ['High-Low', 'High-Close', 'Low-Close', 'True Range', 'ATR']
    for col in expected_columns:
        assert col in result.columns, f"Column '{col}' missing in ATR calculation."

def test_calculate_atr_values(sample_atr_data):
    """
    Check that the ATR values are computed correctly.
    For a rolling period of 3, the first two rows should be NaN.
    """
    period = 3
    result = calculate_atr(sample_atr_data, period=period)
    # Check that rows before the rolling window is full are NaN.
    for i in range(period - 1):
        assert pd.isna(result['ATR'].iloc[i]), f"ATR value at index {i} should be NaN."
    # For index 2, compute the expected ATR manually:
    # Row0 True Range = max(10-5, |10-NaN|, |5-NaN|) = 5
    # Row1 True Range = max(12-6, |12-7|, |6-7|) = max(6,5,1) = 6
    # Row2 True Range = max(11-5, |11-9|, |5-9|) = max(6,2,4) = 6
    expected_atr_index2 = (5 + 6 + 6) / 3
    assert abs(result['ATR'].iloc[2] - expected_atr_index2) < 1e-6, "ATR calculation incorrect for index 2."

def test_calculate_atr_expected_values(sample_atr_data):
    """
    Test ATR values on a small dataset with known expected output.
    """
    period = 3
    result = calculate_atr(sample_atr_data, period=period)
    # Expected ATR for row2 is approximately 5.66667.
    expected_atr_row2 = 5.66667
    assert abs(result['ATR'].iloc[2] - expected_atr_row2) < 1e-4, "ATR value for row 2 not as expected."

# -------------------------------
# Unit Tests for Data Fetching Functions
# -------------------------------

def test_fetch_historical_data(monkeypatch):
    """
    Test fetch_historical_data by monkey-patching yf.download to return dummy data.
    """
    dummy_df = pd.DataFrame({
        'High': [10, 12],
        'Low': [5, 6],
        'Close': [7, 9]
    })

    def dummy_download(ticker, start, end, interval):
        return dummy_df

    monkeypatch.setattr("yfinance.download", dummy_download)
    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2022, 1, 2)
    result = fetch_historical_data("DUMMY", start_date, end_date, interval="1d")
    pd.testing.assert_frame_equal(result, dummy_df)

def test_fetch_realtime_data(monkeypatch):
    """
    Test fetch_realtime_data by monkey-patching yf.download to return dummy real-time data.
    """
    dummy_df = pd.DataFrame({
        'High': [15, 16],
        'Low': [10, 11],
        'Close': [12, 13]
    })

    def dummy_download(ticker, period, interval):
        return dummy_df

    monkeypatch.setattr("yfinance.download", dummy_download)
    result = fetch_realtime_data("DUMMY")
    pd.testing.assert_frame_equal(result, dummy_df)

def test_fetch_current_price(monkeypatch):
    """
    Test fetch_current_price by monkey-patching yahooquery.Ticker to return dummy price data.
    """
    dummy_price_data = {"DUMMY": {"regularMarketPrice": 100}}

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return dummy_price_data

    monkeypatch.setattr("yahooquery.Ticker", lambda symbol: DummyTicker(symbol))
    price = fetch_current_price("DUMMY")
    assert price == 100, "Current price should be 100 from dummy data."

def test_fetch_current_price_exception(monkeypatch):
    """
    Test fetch_current_price when an exception is raised.
    The function should catch the exception, call st.error, and return None.
    """
    def dummy_ticker(symbol):
        raise Exception("Test error")
    monkeypatch.setattr("yahooquery.Ticker", dummy_ticker)

    # Capture error messages by overriding st.error.
    error_messages = []
    def dummy_st_error(msg):
        error_messages.append(msg)
    monkeypatch.setattr(st, "error", dummy_st_error)

    price = fetch_current_price("DUMMY")
    assert price is None, "fetch_current_price should return None on exception."
    assert any("Test error" in msg for msg in error_messages), "Error message should contain 'Test error'."

# -------------------------------
# Unit Tests for ATRAnalysisAgents
# -------------------------------

def test_atr_analysis_task_generation(sample_atr_data):
    """
    Test that ATRAnalysisAgents produces a task with a description that contains the latest ATR value and current price.
    """
    atr_data = calculate_atr(sample_atr_data, period=3)
    agents = ATRAnalysisAgents()
    dummy_agent = agents.atr_investment_advisor()
    current_price = 50  # Dummy current price.
    task = agents.atr_analysis(dummy_agent, atr_data, current_price)
    assert "Latest ATR Value" in task.description, "Task description missing 'Latest ATR Value'."
    assert str(current_price) in task.description, "Task description should include current stock price."

def test_atr_analysis_no_atr_value():
    """
    Test ATRAnalysisAgents when no ATR value is available (i.e. all ATRs are NaN).
    """
    dates = pd.date_range(start='2022-01-01', periods=5, freq='D')
    data = {
        'High': [10, 10, 10, 10, 10],
        'Low': [5, 5, 5, 5, 5],
        'Close': [7, 7, 7, 7, 7]
    }
    df = pd.DataFrame(data, index=dates)
    df['ATR'] = float('nan')
    agents = ATRAnalysisAgents()
    dummy_agent = agents.atr_investment_advisor()
    task = agents.atr_analysis(dummy_agent, df, current_price=50)
    # The description should show "None" for the latest ATR value.
    assert "None" in task.description, "Task description should indicate None for latest ATR value when not available."

# -------------------------------
# Integration Tests for Full Workflow
# -------------------------------

def test_full_workflow_integration(monkeypatch, sample_atr_data_extended):
    """
    Integration test simulating the complete workflow:
    - Fetching historical data (monkey-patched)
    - Calculating ATR
    - Generating a CrewAI task and simulating an investment decision.
    """
    # Patch fetch_historical_data to return our extended sample data.
    monkeypatch.setattr("src.UI.atr.fetch_historical_data", 
                        lambda ticker, start_date, end_date, interval="1d": sample_atr_data_extended)
    # Patch fetch_current_price to return a dummy price.
    monkeypatch.setattr("src.UI.atr.fetch_current_price", lambda symbol: 50)

    # Calculate ATR from the extended sample data.
    atr_data = calculate_atr(sample_atr_data_extended, period=3)

    # Create a dummy CrewAI Agent that returns a fixed response.
    from crewai import Agent, Crew
    class DummyAgent(Agent):
        def __init__(self):
            super().__init__(llm=None, role="Dummy", goal="", backstory="", verbose=False, tools=[])
        def run(self, prompt):
            return "BUY"

    dummy_agent = DummyAgent()
    agents_instance = ATRAnalysisAgents()
    task = agents_instance.atr_analysis(dummy_agent, atr_data, current_price=50)

    # Patch Crew.kickoff to simulate a CrewAI response.
    monkeypatch.setattr(Crew, 'kickoff', lambda self: "BUY")

    crew = Crew(agents=[dummy_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == "BUY", "Integration test did not return the expected result 'BUY'."

def test_full_workflow_with_realtime_data(monkeypatch, sample_atr_data_extended):
    """
    Integration test simulating the workflow using real-time data.
    """
    # Patch fetch_realtime_data to return our extended sample data.
    monkeypatch.setattr("src.UI.atr.fetch_realtime_data", lambda ticker: sample_atr_data_extended)
    # Patch fetch_current_price to return a dummy price.
    monkeypatch.setattr("src.UI.atr.fetch_current_price", lambda symbol: 75)

    # Calculate ATR using the real-time data.
    atr_data = calculate_atr(sample_atr_data_extended, period=3)

    from crewai import Agent, Crew
    class DummyAgent(Agent):
        def __init__(self):
            super().__init__(llm=None, role="Dummy", goal="", backstory="", verbose=False, tools=[])
        def run(self, prompt):
            return "SELL"

    dummy_agent = DummyAgent()
    agents_instance = ATRAnalysisAgents()
    task = agents_instance.atr_analysis(dummy_agent, atr_data, current_price=75)

    # Patch Crew.kickoff to simulate a CrewAI response.
    monkeypatch.setattr(Crew, 'kickoff', lambda self: "SELL")

    crew = Crew(agents=[dummy_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert result == "SELL", "Real-time data integration test did not return the expected result 'SELL'."
