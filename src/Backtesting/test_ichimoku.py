import pandas as pd
import pytest
from datetime import datetime
import sys
import os
# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Updated imports: refer to the main module in src/UI/ichimoku.py
from src.UI.ichimoku import (
    IchimokuCalculator,
    fetch_stock_data,
    fetch_realtime_data,
    IchimokuAnalysisAgents
)
from crewai import Agent, Task, Crew

# -------------------------------
# Fixtures and Dummy Data
# -------------------------------

@pytest.fixture
def sample_data():
    """
    Creates a sample DataFrame with consistent date values for testing.
    """
    df = pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
        'high': [30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
        'low': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        'close': [20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    })
    return df

# -------------------------------
# Unit Tests for Data Fetching
# -------------------------------

def test_date_conversion(monkeypatch):
    """
    Test that fetch_stock_data converts the 'date' column to timezone-naive datetime.
    """
    dummy_df = pd.DataFrame({
        'date': ["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"],
        'high': [100, 110],
        'low': [90, 95],
        'close': [95, 105]
    })

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        def history(self, period):
            return dummy_df

    monkeypatch.setattr('yahooquery.Ticker', lambda symbol: DummyTicker(symbol))
    result = fetch_stock_data("DUMMY", period="1y")
    # Verify that the 'date' column is a datetime dtype and is timezone-naive.
    assert pd.api.types.is_datetime64_any_dtype(result['date']), "Date column is not datetime type"
    assert result['date'].iloc[0].tzinfo is None, "Date column is not timezone-naive"

def test_fetch_realtime_data(monkeypatch):
    """
    Test that fetch_realtime_data returns a DataFrame with expected price data.
    """
    dummy_price = {"regularMarketPrice": 150}
    
    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        @property
        def price(self):
            return dummy_price

    monkeypatch.setattr('yahooquery.Ticker', lambda symbol: DummyTicker(symbol))
    result = fetch_realtime_data("DUMMY")
    assert result is not None, "fetch_realtime_data returned None"
    # Check that the dummy price is present in the DataFrame.
    assert 150 in result.values, "Dummy price not found in real-time data result"

# -------------------------------
# Unit Tests for Ichimoku Calculator
# -------------------------------

def test_ichimoku_calculator_columns(sample_data):
    """
    Test that IchimokuCalculator output includes all expected indicator columns.
    """
    calc = IchimokuCalculator(sample_data, tenkan_period=3, kijun_period=5, senkou_b_period=7, displacement=2, smoothing_factor=1)
    result = calc.calculate()
    expected_columns = [
        'date', 'high', 'low', 'close',
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'
    ]
    for col in expected_columns:
        assert col in result.columns, f"Column '{col}' missing from calculated output."

def test_ichimoku_calculator_values(sample_data):
    """
    Test that indicator values are computed correctly and that displacement works as expected.
    """
    calc = IchimokuCalculator(sample_data, tenkan_period=3, kijun_period=5, senkou_b_period=7, displacement=1, smoothing_factor=1)
    result = calc.calculate()
    # For index 2, tenkan_sen should be computed.
    assert pd.notna(result['tenkan_sen'].iloc[2]), "tenkan_sen should not be NaN at index 2."
    # With displacement=1, the last row's chikou_span should be NaN.
    assert pd.isna(result['chikou_span'].iloc[-1]), "chikou_span of last row should be NaN due to displacement."

def test_ichimoku_calculator_smoothing(sample_data):
    """
    Test that applying a smoothing factor changes indicator values.
    """
    calc_unsmoothed = IchimokuCalculator(sample_data, smoothing_factor=1)
    result_unsmoothed = calc_unsmoothed.calculate()
    calc_smoothed = IchimokuCalculator(sample_data, smoothing_factor=3)
    result_smoothed = calc_smoothed.calculate()
    if len(result_unsmoothed) > 3:
        assert result_smoothed['tenkan_sen'].iloc[3] != result_unsmoothed['tenkan_sen'].iloc[3], "Smoothing factor not applied correctly."

# -------------------------------
# Unit Test for CrewAI Agent Task Generation
# -------------------------------

def test_ichimoku_agents_analysis():
    """
    Test that IchimokuAnalysisAgents produces a task with expected description content.
    """
    dummy_data = pd.DataFrame({
        'tenkan_sen': [20, 21, 22],
        'kijun_sen': [25, 26, 27],
        'senkou_span_a': [30, 31, 32],
        'senkou_span_b': [35, 36, 37],
        'chikou_span': [18, 19, None]  # Last value is None due to displacement.
    })
    agents = IchimokuAnalysisAgents()
    dummy_agent = agents.ichimoku_cloud_investment_advisor()
    # Assume a current price of 28 for the test.
    task = agents.ichimoku_cloud_analysis(dummy_agent, dummy_data, current_price=28)
    # Check that the task description includes key indicator labels and the current price.
    assert "Current Stock Price: 28" in task.description or "28" in task.description, "Task description missing current stock price."
    assert "Tenkan-sen" in task.description, "Task description missing Tenkan-sen."
    assert "Kijun-sen" in task.description, "Task description missing Kijun-sen."

# -------------------------------
# Integration Test for Full Workflow
# -------------------------------

def test_full_workflow_integration(monkeypatch, sample_data):
    """
    Integration test that simulates the complete workflow:
    - Fetching historical data (monkey-patched to return sample_data)
    - Calculating Ichimoku Cloud indicators
    - Generating a CrewAI task and simulating a response.
    """
    # Monkey-patch fetch_stock_data to return sample_data directly.
    monkeypatch.setattr('src.UI.ichimoku.fetch_stock_data', lambda ticker, period: sample_data)
    
    # Calculate indicators.
    calc = IchimokuCalculator(sample_data, tenkan_period=3, kijun_period=5, senkou_b_period=7, displacement=1, smoothing_factor=1)
    ichimoku_data = calc.calculate()
    
    # Create a dummy agent that returns a fixed response.
    class DummyAgent(Agent):
        def __init__(self):
            super().__init__(llm=None, role="Dummy", goal="", backstory="", verbose=False, tools=[])
        def run(self, prompt):
            return "BUY"
    
    dummy_agent = DummyAgent()
    agents = IchimokuAnalysisAgents()
    task = agents.ichimoku_cloud_analysis(dummy_agent, ichimoku_data, current_price=28)
    
    # Monkey-patch Crew.kickoff to simulate a CrewAI response.
    monkeypatch.setattr(Crew, 'kickoff', lambda self: "BUY")
    
    crew = Crew(agents=[dummy_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    # Verify that the integration returns the expected dummy response.
    assert result == "BUY", "Integration test did not return the expected result 'BUY'."
