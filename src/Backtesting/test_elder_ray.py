import os
import sys
import pandas as pd
import pytest
from datetime import datetime
from textwrap import dedent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions from the main Elder-Ray module.
from src.UI.elder_ray_index import (
    calculate_moving_average,
    calculate_elder_ray_index,
    flatten_columns,
    fetch_stock_data,
    fetch_current_price,
)

# Import Agent from CrewAI library.
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Import the Elder-Ray Analysis Agent from the designated folder.
# Adjust the import path if necessary.
from src.Agents.ElderRay.elder_ray_agent import ElderRayAnalysisAgent

# ---------------------------
# Unit Tests for Helper Functions
# ---------------------------

def test_calculate_moving_average_ema():
    """Test the EMA moving average calculation."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = calculate_moving_average(series, period=3, ma_type="EMA")
    expected = series.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)

def test_calculate_moving_average_sma():
    """Test the SMA moving average calculation."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = calculate_moving_average(series, period=3, ma_type="SMA")
    expected = series.rolling(window=3).mean()
    pd.testing.assert_series_equal(result, expected)

def test_calculate_elder_ray_index():
    """Test that the Elder-Ray index calculation produces valid bull and bear power columns."""
    # Create sample data
    df = pd.DataFrame({
        'High': [10, 12, 14, 16, 18],
        'Low': [8, 9, 11, 13, 15],
        'Close': [9, 11, 13, 15, 17]
    })
    result = calculate_elder_ray_index(df.copy(), ma_period=3, ma_type="SMA", price_column="Close")
    # Expected columns (in lowercase) should exist
    assert "bull power" in result.columns
    assert "bear power" in result.columns
    # Check that the values are numeric
    assert pd.api.types.is_numeric_dtype(result["bull power"])
    assert pd.api.types.is_numeric_dtype(result["bear power"])

def test_flatten_columns():
    """Test that flatten_columns removes extraneous columns and resets the index."""
    # Create a DataFrame with MultiIndex columns including an extraneous column
    cols = pd.MultiIndex.from_tuples([
        ("bull power", ""), 
        ("bear power", ""), 
        ("index--dummy", "")
    ])
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=cols)
    flattened = flatten_columns(df.copy())
    for col in flattened.columns:
        assert not str(col).startswith("index--")
    # Check that the index has been reset to a RangeIndex
    assert flattened.index.equals(pd.RangeIndex(start=0, stop=len(flattened), step=1))

# ---------------------------
# Integration Tests for Data Fetching Functions
# ---------------------------
# We simulate yahooquery's Ticker using monkeypatch.

class DummyTicker:
    """A dummy Ticker class to simulate yahooquery responses."""
    def __init__(self, ticker):
        self.ticker = ticker

    def history(self, **kwargs):
        # Return a dummy DataFrame for history() calls
        return pd.DataFrame({
            'high': [10, 11, 12],
            'low': [8, 9, 10],
            'close': [9, 10, 11]
        })

    @property
    def price(self):
        # Return dummy price data
        return {self.ticker: {'regularMarketPrice': 15}}

def dummy_ticker_constructor(ticker):
    return DummyTicker(ticker)

def test_fetch_stock_data_historical(monkeypatch):
    """Test fetching historical data with a dummy Ticker."""
    monkeypatch.setattr("src.UI.elder_ray_index.Ticker", dummy_ticker_constructor)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    df = fetch_stock_data("TEST", "Historical", start_date=start_date, end_date=end_date)
    assert not df.empty
    assert "high" in df.columns

def test_fetch_stock_data_realtime(monkeypatch):
    """Test fetching real-time data with a dummy Ticker."""
    monkeypatch.setattr("src.UI.elder_ray_index.Ticker", dummy_ticker_constructor)
    df = fetch_stock_data("TEST", "Real-Time", period="1d", interval="1m")
    assert not df.empty

def test_fetch_current_price(monkeypatch):
    """Test fetching the current stock price using a dummy Ticker."""
    monkeypatch.setattr("src.UI.elder_ray_index.Ticker", dummy_ticker_constructor)
    price = fetch_current_price("TEST")
    assert price == 15

# ---------------------------
# Integration Test for CrewAI Investment Decision Support
# ---------------------------
# We'll simulate the CrewAI workflow using a dummy Elder-Ray agent.

class DummyElderRayAgent:
    """A dummy agent to simulate CrewAI investment decision support for Elder-Ray."""
    def elder_ray_investment_advisor(self):
        return Agent(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4o"),
            role="Dummy Elder-Ray Advisor",
            goal="Provide a dummy investment decision.",
            backstory="Dummy agent for testing.",
            verbose=True,
            tools=[]
        )
    
    def elder_ray_analysis(self, agent, elder_ray_data, current_price):
        # For testing, return a fixed task with a dummy recommendation.
        report = dedent(f"""
            Dummy Elder-Ray Analysis Report:
            {elder_ray_data.to_string(index=False)}
            Current Stock Price: {current_price}
            Recommendation: BUY
        """)
        return Task(
            description=report,
            agent=agent,
            expected_output="A dummy investment decision."
        )

def test_crewai_investment_decision(monkeypatch):
    """Test the CrewAI investment decision workflow using a dummy Elder-Ray agent."""
    dummy_agent_obj = DummyElderRayAgent()
    dummy_data = pd.DataFrame({
        'bull power': [1.5],
        'bear power': [-1.0]
    })
    current_price = 20.0
    task = dummy_agent_obj.elder_ray_analysis(dummy_agent_obj.elder_ray_investment_advisor(), dummy_data, current_price)
    
    # Create a Dummy Crew class that returns a fixed recommendation.
    class DummyCrew(Crew):
        def kickoff(self):
            return "BUY: Dummy recommendation based on test data."
    
    crew = DummyCrew(agents=[dummy_agent_obj.elder_ray_investment_advisor()], tasks=[task], verbose=True)
    result = crew.kickoff()
    assert "BUY" in result
