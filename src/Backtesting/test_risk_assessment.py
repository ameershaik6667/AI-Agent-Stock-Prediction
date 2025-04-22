# tests/test_risk_assessment.py

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------------------------------------------------
# Ensure we can import src/UI/risk_assessment.py as risk_dashboard
# -------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import src.UI.risk_assessment as risk_dashboard  # noqa: E402

# -------------------------------------------------------------------
# Automatically stub out st.info, st.error, and st.pyplot
# -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def stub_streamlit(monkeypatch):
    class DummySt:
        @staticmethod
        def info(*args, **kwargs): pass
        @staticmethod
        def error(*args, **kwargs): pass
        @staticmethod
        def pyplot(fig): pass

    monkeypatch.setattr(risk_dashboard, 'st', DummySt)
    yield

# -------------------------------------------------------------------
# Unit Tests: fetch_stock_data
# -------------------------------------------------------------------
def test_fetch_stock_data_success(monkeypatch):
    dates = pd.date_range('2021-01-01', periods=3, freq='D', tz='UTC')
    hist = pd.DataFrame({
        'date': dates,
        'high': [10, 20, 30],
        'low': [5, 15, 25],
        'close': [7, 17, 27],
    }).set_index('date')

    class DummyTicker:
        def __init__(self, symbol): pass
        def history(self, period): return hist

    monkeypatch.setattr(risk_dashboard, 'Ticker', DummyTicker)
    df = risk_dashboard.fetch_stock_data('TEST', period='1y')

    # Check DataFrame shape & columns
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns).count('date') == 1
    for col in ['high', 'low', 'close']:
        assert col in df.columns
    # Date should be tz-naive datetime64
    assert df['date'].dtype == 'datetime64[ns]'

def test_fetch_stock_data_missing_column(monkeypatch):
    # Missing 'low' and 'close'
    hist = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=2, tz='UTC'),
        'high': [10, 20],
    }).set_index('date')

    class DummyTicker:
        def __init__(self, symbol): pass
        def history(self, period): return hist

    errors = []
    monkeypatch.setattr(risk_dashboard, 'Ticker', DummyTicker)
    monkeypatch.setattr(risk_dashboard.st, 'error', lambda msg: errors.append(msg))

    df = risk_dashboard.fetch_stock_data('TEST', period='1y')
    assert df is None
    assert any("Required column 'low' not found" in e for e in errors)

def test_fetch_stock_data_non_dataframe(monkeypatch):
    # history returns a non-DataFrame
    class DummyTicker:
        def __init__(self, symbol): pass
        def history(self, period): return "oops"

    errors = []
    monkeypatch.setattr(risk_dashboard, 'Ticker', DummyTicker)
    monkeypatch.setattr(risk_dashboard.st, 'error', lambda msg: errors.append(msg))

    df = risk_dashboard.fetch_stock_data('X', period='1y')
    assert df is None
    assert any("Failed to fetch data as a DataFrame" in e for e in errors)

def test_fetch_stock_data_rename_columns(monkeypatch):
    # Columns are capitalized
    hist = pd.DataFrame({
        'Date': pd.date_range('2021-01-01', periods=2, tz='UTC'),
        'High': [10, 20],
        'Low': [5, 15],
        'Close': [7, 17],
    }).set_index('Date')

    class DummyTicker:
        def __init__(self, symbol): pass
        def history(self, period): return hist

    monkeypatch.setattr(risk_dashboard, 'Ticker', DummyTicker)
    df = risk_dashboard.fetch_stock_data('X', period='1y')

    # After renaming, lowercase columns should exist
    for col in ['date', 'high', 'low', 'close']:
        assert col in df.columns

# -------------------------------------------------------------------
# Unit Test: fetch_current_stock_price
# -------------------------------------------------------------------
def test_fetch_current_stock_price_success(monkeypatch):
    class DummyTicker:
        def __init__(self, symbol):
            self.price = {symbol.upper(): {'regularMarketPrice': 99.99}}

    monkeypatch.setattr(risk_dashboard, 'Ticker', DummyTicker)
    price = risk_dashboard.fetch_current_stock_price('test')
    assert isinstance(price, float)
    assert price == 99.99

def test_fetch_current_stock_price_error(monkeypatch):
    # price dict missing regularMarketPrice
    class DummyTicker:
        def __init__(self, symbol):
            self.price = {symbol.upper(): {}}

    errors = []
    monkeypatch.setattr(risk_dashboard, 'Ticker', DummyTicker)
    monkeypatch.setattr(risk_dashboard.st, 'error', lambda msg: errors.append(msg))

    price = risk_dashboard.fetch_current_stock_price('XYZ')
    assert price is None
    assert any("Failed to fetch current stock price" in e for e in errors)

# -------------------------------------------------------------------
# Unit Tests: Risk Metrics Calculations
# -------------------------------------------------------------------
def test_calculate_risk_metrics_basic():
    data = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=5),
        'close': [100, 110, 105, 115, 120],
    })
    metrics, updated = risk_dashboard.calculate_risk_metrics(data, confidence=0.05)

    assert set(metrics) == {'var', 'max_drawdown', 'volatility'}
    # VaR matches numpy percentile
    returns = updated['returns'].dropna().values
    expected = np.percentile(returns, 5)
    assert pytest.approx(expected, rel=1e-6) == metrics['var']

    # Check max_drawdown: if returns always positive, drawdown >= 0
    assert updated['drawdown'].min() >= 0

def test_calculate_scenario_risk_metrics_basic():
    data = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=4),
        'close': [100, 105, 95, 100],
    })
    base, _ = risk_dashboard.calculate_risk_metrics(data, confidence=0.1)
    scenario = risk_dashboard.calculate_scenario_risk_metrics(data, shock=0.05, confidence=0.1)

    assert set(scenario) == {'var', 'max_drawdown', 'volatility'}
    # Under a negative shock, VaR should be at least as bad
    assert scenario['var'] <= base['var']

# -------------------------------------------------------------------
# Integration Test: Portfolio Breakdown
# -------------------------------------------------------------------
def test_analyze_portfolio_breakdown_single_class(monkeypatch):
    # All metrics constant
    monkeypatch.setattr(risk_dashboard, 'fetch_stock_data',
                        lambda t, p: pd.DataFrame({'date': pd.date_range('2021-01-01', 3),
                                                   'close': [100, 100, 100]}))
    monkeypatch.setattr(risk_dashboard, 'calculate_risk_metrics',
                        lambda df, c: ({'var': -0.02, 'max_drawdown': -0.03, 'volatility': 0.1}, df))

    details, grouped = risk_dashboard.analyze_portfolio_breakdown(
        "AAA, Equity, 100\nBBB, Equity, 300"
    )
    # Two positions, one group
    assert len(details) == 2
    assert len(grouped) == 1
    assert grouped.loc[0, 'Weighted VaR'] == -0.02

def test_analyze_portfolio_breakdown_multiple_classes(monkeypatch):
    # Stub data & metrics
    monkeypatch.setattr(risk_dashboard, 'fetch_stock_data',
                        lambda t, p: pd.DataFrame({'date': pd.date_range('2021-01-01', 2),
                                                   'close': [100, 100]}))
    monkeypatch.setattr(risk_dashboard, 'calculate_risk_metrics',
                        lambda df, c: ({'var': -0.01, 'max_drawdown': -0.02, 'volatility': 0.05}, df))

    details, grouped = risk_dashboard.analyze_portfolio_breakdown(
        "AAA, Equity, 100\nCCC, Bond, 200"
    )
    # Two positions, two groups
    assert len(details) == 2
    assert set(grouped['Asset Class']) == {'Equity', 'Bond'}
    # Each group has its own weighted VaR (same here)
    assert all(grouped['Weighted VaR'] == -0.01)

# -------------------------------------------------------------------
# Plotting Functions Tests
# -------------------------------------------------------------------
def test_plot_price_chart(monkeypatch):
    calls = []
    monkeypatch.setattr(risk_dashboard.st, 'pyplot', lambda fig: calls.append(fig))

    df = pd.DataFrame({
        'date': [datetime(2021,1,1), datetime(2021,1,2)],
        'close': [1.0, 2.0]
    })
    risk_dashboard.plot_price_chart(df, 'TST')

    assert len(calls) == 1
    fig = calls[0]
    ax = fig.axes[0]
    line = ax.lines[0]
    # Verify label & data
    assert line.get_label() == 'Close Price'
    assert list(line.get_ydata()) == [1.0, 2.0]

def test_plot_drawdown_chart(monkeypatch):
    calls = []
    monkeypatch.setattr(risk_dashboard.st, 'pyplot', lambda fig: calls.append(fig))

    df = pd.DataFrame({
        'date': [datetime(2021,1,1), datetime(2021,1,2)],
        'drawdown': [0.0, -0.5]
    })
    risk_dashboard.plot_drawdown_chart(df, 'TST')

    assert calls
    ax = calls[0].axes[0]
    line = ax.lines[0]
    assert line.get_label() == 'Drawdown'
    assert list(line.get_ydata()) == [0.0, -0.5]

def test_plot_return_histogram(monkeypatch):
    calls = []
    monkeypatch.setattr(risk_dashboard.st, 'pyplot', lambda fig: calls.append(fig))

    returns = pd.Series([0.01, -0.02, 0.03])
    var_val = -0.02
    risk_dashboard.plot_return_histogram(returns, var_val)

    assert calls
    ax = calls[0].axes[0]
    # The first line in ax.lines is the VaR vertical line
    v_line = ax.lines[0]
    # All x-points on that line equal var_val
    xs = v_line.get_xdata()
    assert all(np.isclose(xs, var_val))

