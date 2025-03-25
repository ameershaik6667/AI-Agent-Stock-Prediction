import os
import sys
import pytest
import pandas as pd
import numpy as np
from textwrap import dedent

# Import the functions to be tested from your module.
# Adjust the import path as needed.
# Update the system path to import modules from parent directories if needed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.UI.gann_main import calculate_gann_hi_lo_activator, fetch_historical_data, fetch_realtime_data, fetch_current_price, GannAnalysisAgents

# -------------------------
# Unit Tests for Gann Activator
# -------------------------

@pytest.fixture
def sample_data_no_smoothing():
    """
    Creates a sample DataFrame with known values to test the Gann Hi-Lo Activator calculation.
    In this sample, we design rows that trigger both branches of the condition.
    """
    # Construct a DataFrame with columns: Low, High, Close.
    # Using 5 rows with conditions:
    # Row0: activator = Low0 = 100
    # Row1: Close 103 > 100 so activator = min(Low1, 100) = min(99, 100)= 99
    # Row2: Close 102 > 99 so activator = min(Low2, 99) = min(98, 99)= 98
    # Row3: Close 96 <= 98 so activator = max(High3, 98) = max(102, 98)= 102
    # Row4: Close 99 <= 102 so activator = max(High4, 102) = max(101, 102)= 102
    data = pd.DataFrame({
        'Low': [100, 99, 98, 97, 96],
        'High': [105, 104, 103, 102, 101],
        'Close': [102, 103, 102, 96, 99]
    })
    return data

def test_calculate_gann_hi_lo_activator_no_smoothing(sample_data_no_smoothing):
    """
    Test the Gann Hi-Lo Activator function with a smoothing period of 1 (i.e., no smoothing).
    We verify that the raw activator values are as expected.
    """
    # With smoothing_period=1, no EMA smoothing is applied.
    result = calculate_gann_hi_lo_activator(sample_data_no_smoothing.copy(), smoothing_period=1)
    expected_raw = [100, 99, 98, 102, 102]
    # Compare raw values
    np.testing.assert_array_almost_equal(result['Gann Hi Lo'].values, np.array(expected_raw))
    # When no smoothing is applied, smoothed equals raw
    np.testing.assert_array_almost_equal(result['Gann Hi Lo Smoothed'].values, np.array(expected_raw))

def test_calculate_gann_hi_lo_activator_with_smoothing(sample_data_no_smoothing):
    """
    Test the Gann Hi-Lo Activator function with smoothing applied (smoothing_period > 1).
    We check that the smoothed column is computed and not identical to the raw column.
    """
    result = calculate_gann_hi_lo_activator(sample_data_no_smoothing.copy(), smoothing_period=3)
    raw = result['Gann Hi Lo'].values
    smoothed = result['Gann Hi Lo Smoothed'].values
    # Ensure that smoothing alters the values (they may be similar but not exactly equal)
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(raw, smoothed)
    # Also, verify that the smoothed series is of the same length
    assert len(raw) == len(smoothed)

# -------------------------
# Integration Tests for Data Retrieval Functions
# -------------------------
class DummyTicker:
    """A dummy Ticker class to simulate yahooquery.Ticker behavior."""
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        # Return a simple DataFrame with required columns
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=3, freq='D'),
            'high': [110, 112, 111],
            'low': [100, 101, 99],
            'close': [105, 107, 106]
        })
        return df

    @property
    def price(self):
        # Return a dictionary mimicking a real ticker.price response
        return {self.symbol: {'regularMarketPrice': 107.5}}

@pytest.fixture
def monkey_ticker(monkeypatch):
    """
    Fixture to monkeypatch the Ticker class in the module to use DummyTicker.
    """
    def dummy_ticker(symbol):
        return DummyTicker(symbol)
    monkeypatch.setattr("src.UI.gann_main.Ticker", dummy_ticker)

def test_fetch_historical_data(monkey_ticker):
    from src.UI.gann_main import fetch_historical_data
    data = fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
    assert not data.empty
    # Check that required columns exist
    for col in ['date', 'high', 'low', 'close']:
        assert col in data.columns

def test_fetch_realtime_data(monkey_ticker):
    from src.UI.gann_main import fetch_realtime_data
    data = fetch_realtime_data("AAPL")
    assert not data.empty
    assert 'date' in data.columns

def test_fetch_current_price(monkey_ticker):
    from src.UI.gann_main import fetch_current_price
    price = fetch_current_price("AAPL")
    assert price == 107.5

# -------------------------
# Integration Test for CrewAI Investment Decision Task
# -------------------------
def test_gann_analysis_task(monkey_ticker):
    """
    Test that the CrewAI investment decision task is generated correctly,
    including the latest Gann values and current stock price.
    """
    # Create a dummy Gann data DataFrame
    gann_data = pd.DataFrame({
        'Gann Hi Lo': [100, 101, 102],
        'Gann Hi Lo Smoothed': [100, 101, 102]
    })
    # Use a dummy current price
    current_price = 105.0

    agents = GannAnalysisAgents()
    advisor_agent = agents.gann_investment_advisor()
    task = agents.gann_analysis(advisor_agent, gann_data, current_price)

    # Check that the task description includes the expected values
    assert "Raw Value: 102" in task.description
    assert "Smoothed Value: 102" in task.description
    assert f"Current Stock Price: {current_price}" in task.description
    # Check expected output string
    assert "Investment decision" in task.expected_output

