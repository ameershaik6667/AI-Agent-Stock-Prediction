import pytest
import pandas as pd
import numpy as np
from textwrap import dedent
import streamlit as st
import yfinance as yf
import sys
import os

# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions and classes from the CCI app module.
from src.UI.cci import (
    flatten_columns,
    standardize_columns,
    calculate_cci,
    fetch_stock_data,
    fetch_current_price,
    CCIAnalysisAgents,
    main
)

# -------------------------------------------
# Fixtures and Dummy Data
# -------------------------------------------
@pytest.fixture
def sample_stock_data():
    """
    Creates a sample DataFrame for stock data testing.
    """
    df = pd.DataFrame({
        "Date": pd.date_range(start="2022-01-01", periods=5, freq="D"),
        "High": [110, 112, 111, 115, 116],
        "Low": [100, 102, 101, 105, 107],
        "Close": [105, 108, 107, 110, 112],
        "Open": [104, 107, 106, 109, 111],
        "Volume": [1000, 1050, 1100, 1150, 1200]
    })
    # Convert columns to lower-case to match expected format.
    df.columns = [col.lower() for col in df.columns]
    return df

# -------------------------------------------
# Unit Tests for Helper Functions
# -------------------------------------------
def test_flatten_columns():
    # Create a DataFrame with MultiIndex columns.
    arrays = [["AAPL", "AAPL"], ["High", "Low"]]
    mi = pd.MultiIndex.from_arrays(arrays)
    df = pd.DataFrame([[110, 100], [112, 102]], columns=mi)
    flattened = flatten_columns(df)
    expected = ["AAPL High", "AAPL Low"]
    assert list(flattened.columns) == expected

def test_standardize_columns():
    df = pd.DataFrame(columns=["high AAPL", "low AAPL", "close AAPL"])
    standardized = standardize_columns(df)
    expected = ["high", "low", "close"]
    assert list(standardized.columns) == expected

def test_calculate_cci(sample_stock_data):
    # Calculate CCI with a rolling period of 3.
    cci_series = calculate_cci(sample_stock_data, period=3)
    # Ensure the output has the same number of rows.
    assert len(cci_series) == len(sample_stock_data)
    # Ensure that after the rolling window, some values are not NaN.
    assert cci_series.dropna().shape[0] > 0

# -------------------------------------------
# Unit Tests for Data Fetching Functions
# -------------------------------------------
def dummy_download(ticker, period, interval):
    """
    Dummy function to simulate yf.download.
    """
    return pd.DataFrame({
        "Open": [100, 101, 102],
        "High": [105, 106, 107],
        "Low": [95, 96, 97],
        "Close": [102, 103, 104],
        "Volume": [1000, 1100, 1200]
    })

def test_fetch_stock_data(monkeypatch):
    monkeypatch.setattr(yf, "download", dummy_download)
    df = fetch_stock_data("AAPL", period="1y", interval="1d")
    assert not df.empty
    # Check that expected columns (case-insensitive) are present.
    cols = [col.lower() for col in df.columns]
    for col in ["high", "low", "close"]:
        assert col in cols

class DummyTicker:
    """
    Dummy Ticker class to simulate yahooquery.Ticker.
    """
    def __init__(self, symbol):
        self.symbol = symbol
    @property
    def price(self):
        return {self.symbol: {"regularMarketPrice": 123.45}}

def test_fetch_current_price(monkeypatch):
    # Monkeypatch Ticker in the src.UI.cci module.
    monkeypatch.setattr("src.UI.cci.Ticker", lambda symbol: DummyTicker(symbol))
    price = fetch_current_price("AAPL")
    assert price == 123.45

# -------------------------------------------
# Unit Test for CrewAI Agent Task Generation
# -------------------------------------------
def test_cci_aggregate_analysis():
    agents_instance = CCIAnalysisAgents()
    advisor = agents_instance.cci_investment_advisor()
    sentiment = agents_instance.cci_sentiment_analyst()
    research = agents_instance.cci_research_analyst()
    agents = [advisor, sentiment, research]
    tasks = agents_instance.cci_aggregate_analysis(agents, cci_value=50.0, current_price=150.0)
    # Verify that one task is created per agent and the description includes the provided values.
    assert len(tasks) == 3
    for task in tasks:
        assert "50.0" in task.description
        assert "150.0" in task.description

# -------------------------------------------
# Integration Tests
# -------------------------------------------
def test_full_pipeline_integration(monkeypatch, sample_stock_data):
    """
    Integration test for the complete pipeline:
    - Simulate fetching stock data,
    - Calculate CCI,
    - And populate session_state.
    """
    monkeypatch.setattr(yf, "download", lambda ticker, period, interval: sample_stock_data)
    data = fetch_stock_data("AAPL", period="1y", interval="1d")
    assert not data.empty

    cci_series = calculate_cci(data, period=3)
    assert cci_series.dropna().shape[0] > 0

    # Simulate Streamlit session_state.
    st.session_state = {}
    st.session_state['stock_data'] = data
    try:
        latest_cci = cci_series.dropna().iloc[-1]
        st.session_state['latest_cci'] = latest_cci
    except Exception:
        pytest.fail("Failed to extract latest CCI value.")
    assert "stock_data" in st.session_state
    assert "latest_cci" in st.session_state

def test_investment_decision_integration(monkeypatch, sample_stock_data):
    """
    Integration test for the investment decision branch:
    - Pre-set session_state with stock data and latest CCI.
    - Monkeypatch fetch_current_price and Crew.kickoff.
    """
    st.session_state = {}
    st.session_state['stock_data'] = sample_stock_data
    st.session_state['latest_cci'] = 50.0

    monkeypatch.setattr("src.UI.cci.fetch_current_price", lambda symbol: 150.0)
    dummy_results = ["BUY", "HOLD", "SELL"]
    monkeypatch.setattr("src.UI.cci.Crew.kickoff", lambda self: dummy_results)

    current_price = fetch_current_price("AAPL")
    assert current_price == 150.0

    latest_cci = st.session_state['latest_cci']
    agents_instance = CCIAnalysisAgents()
    advisor = agents_instance.cci_investment_advisor()
    sentiment = agents_instance.cci_sentiment_analyst()
    research = agents_instance.cci_research_analyst()
    agents = [advisor, sentiment, research]
    tasks = agents_instance.cci_aggregate_analysis(agents, latest_cci, current_price)
    crew = Crew(agents=agents, tasks=tasks, verbose=True)
    results = crew.kickoff()
    assert results == dummy_results

def test_realtime_mode_integration(monkeypatch, sample_stock_data):
    """
    Integration test for real-time mode:
    - Force real-time mode settings,
    - Monkeypatch st_autorefresh,
    - And verify session_state is populated.
    """
    monkeypatch.setattr(yf, "download", lambda ticker, period, interval: sample_stock_data)
    monkeypatch.setattr("src.UI.cci.st_autorefresh", lambda interval, limit, key: None)
    monkeypatch.setattr("src.UI.cci.Ticker", lambda symbol: DummyTicker(symbol))
    
    st.session_state = {}
    data = fetch_stock_data("AAPL", period="1d", interval="1m")
    st.session_state['stock_data'] = data
    cci_series = calculate_cci(data, period=3)
    try:
        latest_cci = cci_series.dropna().iloc[-1]
        st.session_state['latest_cci'] = latest_cci
    except Exception:
        pytest.fail("Failed to extract latest CCI value in real-time mode.")
    assert "stock_data" in st.session_state
    assert "latest_cci" in st.session_state
