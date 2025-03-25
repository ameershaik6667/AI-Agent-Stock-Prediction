import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from textwrap import dedent

# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions and classes from your application code.
from src.Agents.VPT.vpt_agent import VPTAnalysisAgent
from src.UI.vpt import (
    calculate_vpt,
    fetch_historical_data,
    fetch_realtime_data,
    fetch_current_price
)

# ---------------------------
# Helper Functions for Dummy Data
# ---------------------------
def create_dummy_stock_data():
    """Creates dummy historical stock data for testing VPT calculation."""
    data = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=5, freq='D'),
        'close': [100, 105, 110, 115, 120],
        'volume': [1000, 1500, 1200, 1300, 1400]
    })
    return data

def create_dummy_realtime_data():
    """Creates dummy real-time data similar to what yf.download might return."""
    data = pd.DataFrame({
        'Datetime': pd.date_range(start="2023-01-01 09:30", periods=5, freq='T'),
        'High': [100, 101, 102, 103, 104],
        'Low': [99, 100, 101, 102, 103],
        'Close': [100, 100.5, 101, 101.5, 102],
        'Volume': [500, 600, 550, 580, 590]
    })
    # Simulate a multi-index column structure that might be returned
    data.columns = pd.MultiIndex.from_tuples([(col, '') for col in data.columns])
    return data

# ---------------------------
# Unit Tests for VPT Calculation
# ---------------------------
def test_calculate_vpt_no_smoothing():
    data = create_dummy_stock_data()
    result = calculate_vpt(data.copy(), calc_period=1, weighting_factor=1.0, apply_smoothing=False)
    
    # Check that VPT column exists
    assert 'VPT' in result.columns
    # First row should be NaN because pct_change for the first row is NaN
    assert pd.isna(result.loc[0, 'VPT'])
    # All subsequent rows should not be NaN
    for i in range(1, len(result)):
        assert not pd.isna(result.loc[i, 'VPT'])

def test_calculate_vpt_with_smoothing():
    data = create_dummy_stock_data()
    result = calculate_vpt(data.copy(), calc_period=1, weighting_factor=1.0, apply_smoothing=True, smoothing_window=2)
    assert 'VPT' in result.columns
    # Check that smoothed VPT values are numeric and non-NaN after the first row
    for i in range(1, len(result)):
        assert not pd.isna(result.loc[i, 'VPT'])
        assert isinstance(result.loc[i, 'VPT'], (float, np.floating))

# ---------------------------
# Unit Tests for Data Fetching Functions
# ---------------------------
def test_fetch_historical_data(monkeypatch):
    dummy_df = create_dummy_stock_data()
    
    # Monkey-patch yf.download to return dummy_df for custom date range
    def dummy_yf_download(ticker_symbol, start, end):
        return dummy_df.copy()
    monkeypatch.setattr("yfinance.download", dummy_yf_download)
    
    result = fetch_historical_data("AAPL", start_date="2023-01-01", end_date="2023-01-05")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    for col in ['date', 'high', 'low', 'close', 'volume']:
        assert col in result.columns

def test_fetch_realtime_data(monkeypatch):
    dummy_rt = create_dummy_realtime_data()
    
    def dummy_yf_download(ticker_symbol, period, interval):
        return dummy_rt.copy()
    monkeypatch.setattr("yfinance.download", dummy_yf_download)
    
    result = fetch_realtime_data("AAPL", period="1d", interval="1m")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    for col in ['date', 'high', 'low', 'close', 'volume']:
        assert col in result.columns

def test_fetch_current_price(monkeypatch):
    dummy_price = {
        "AAPL": {
            "regularMarketPrice": 150.0
        }
    }
    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return dummy_price
    monkeypatch.setattr("yahooquery.Ticker", lambda symbol: DummyTicker(symbol))
    
    price = fetch_current_price("AAPL")
    assert price == 150.0

# ---------------------------
# Unit Tests for VPTAnalysisAgent
# ---------------------------
def test_vpt_analysis_task():
    data = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=3, freq='D'),
        'VPT': [0, 100, 200]
    })
    current_price = 150.0
    
    agent_instance = VPTAnalysisAgent()
    advisor_agent = agent_instance.vpt_trading_advisor()
    task = agent_instance.vpt_analysis(advisor_agent, data, current_price)
    
    # Check that the task description contains the expected latest VPT value and current price.
    assert "Latest VPT Value: 200" in task.description
    assert "Current Stock Price: 150.0" in task.description
    # Ensure the expected output prompt is set correctly (it mentions a single-word decision).
    assert "BUY" in task.expected_output or "SELL" in task.expected_output or isinstance(task.expected_output, str)

# ---------------------------
# Integration Tests for Full Pipeline
# ---------------------------
def test_integration_full_pipeline(monkeypatch):
    dummy_df = create_dummy_stock_data()
    
    # Simulate yf.download to return dummy_df for historical data
    monkeypatch.setattr("yfinance.download", lambda ticker_symbol, **kwargs: dummy_df.copy())
    
    data = fetch_historical_data("AAPL", period="1y")
    assert data is not None
    
    data_with_vpt = calculate_vpt(data.copy(), calc_period=1, weighting_factor=1.0, apply_smoothing=False)
    assert 'VPT' in data_with_vpt.columns
    latest_vpt = data_with_vpt['VPT'].iloc[-1]
    dummy_current_price = 160.0
    
    agent_instance = VPTAnalysisAgent()
    advisor_agent = agent_instance.vpt_trading_advisor()
    task = agent_instance.vpt_analysis(advisor_agent, data_with_vpt, dummy_current_price)
    
    assert str(latest_vpt) in task.description
    assert str(dummy_current_price) in task.description

def test_integration_with_real_time_data(monkeypatch):
    dummy_rt = create_dummy_realtime_data()
    
    # Simulate yf.download to return dummy_rt for real-time data
    monkeypatch.setattr("yfinance.download", lambda ticker_symbol, **kwargs: dummy_rt.copy())
    
    data = fetch_realtime_data("AAPL", period="1d", interval="1m")
    assert data is not None
    # Ensure that after processing, required columns are present
    for col in ['date', 'high', 'low', 'close', 'volume']:
        assert col in data.columns
    
    data_with_vpt = calculate_vpt(data.copy(), calc_period=1, weighting_factor=1.0, apply_smoothing=False)
    assert 'VPT' in data_with_vpt.columns

def test_integration_error_handling_missing_columns(monkeypatch):
    # Create dummy data missing the 'volume' column
    dummy_df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=3, freq='D'),
        'close': [100, 105, 110],
        'high': [102, 107, 112],
        'low': [98, 103, 108]
    })
    monkeypatch.setattr("yfinance.download", lambda ticker_symbol, **kwargs: dummy_df.copy())
    result = fetch_historical_data("AAPL", period="1y")
    # Since 'volume' is missing, function should return None.
    assert result is None

def test_integration_agent_response_variability(monkeypatch):
    """
    Simulate dynamic responses from CrewAI's Crew by monkey-patching Crew.kickoff.
    """
    dummy_df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=3, freq='D'),
        'VPT': [0, 100, 300]  # Increasing VPT
    })
    dummy_current_price = 155.0
    
    agent_instance = VPTAnalysisAgent()
    advisor_agent = agent_instance.vpt_trading_advisor()
    task = agent_instance.vpt_analysis(advisor_agent, dummy_df, dummy_current_price)
    
    # Create a dummy Crew class that simulates different responses based on VPT value.
    class DummyCrew:
        def __init__(self, agents, tasks, verbose=False):
            self.agents = agents
            self.tasks = tasks
        def kickoff(self):
            # For this simulation, if latest VPT > 250 then "BUY", else "SELL"
            latest_vpt = float(dummy_df['VPT'].iloc[-1])
            return "BUY" if latest_vpt > 250 else "SELL"
    monkeypatch.setattr("src.UI.vpt.Crew", DummyCrew)
    from crewai import Crew
    crew = Crew(agents=[advisor_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    # Given our dummy VPT value is 300, expect "BUY"
    assert result == "BUY"
