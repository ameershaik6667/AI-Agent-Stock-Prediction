import pytest
import pandas as pd
import yfinance as yf
from src.UI.nqr import DataFetch, RatioCalc

class DummyTicker:
    def __init__(self, income_df, balance_df):
        self.quarterly_financials = income_df
        self.quarterly_balance_sheet = balance_df

@pytest.fixture(autouse=True)
def patch_yfinance(monkeypatch):
    # Create dummy quarterly_financials: index = ["Total Revenue","Net Income"], columns = [pd.Timestamp]
    date = pd.Timestamp("2023-03-31")
    inc_df = pd.DataFrame(
        {"Total Revenue": [500], "Net Income": [100]},
        index=[date]
    )
    # Transpose to match yfinance structure: index metrics, columns date
    inc_df = inc_df.transpose()
    # Create dummy quarterly_balance_sheet: index = ["Total Stockholder Equity","Total Liab"], columns = [pd.Timestamp]
    bs_df = pd.DataFrame(
        {"Total Stockholder Equity": [200], "Total Liab": [300]},
        index=[date]
    )
    bs_df = bs_df.transpose()
    def mock_ticker(ticker):
        return DummyTicker(inc_df, bs_df)
    monkeypatch.setattr(yf, 'Ticker', mock_ticker)

def test_fetch_income_statement():
    df = DataFetch(num_quarters=1)
    reports = df.fetch_income_statement('AAPL')
    assert isinstance(reports, list)
    assert reports[0]['fiscalDateEnding'] == '2023-03-31'
    assert reports[0]['totalRevenue'] == 500
    assert reports[0]['netIncome'] == 100

def test_fetch_balance_sheet():
    df = DataFetch(num_quarters=1)
    reports = df.fetch_balance_sheet('AAPL')
    assert isinstance(reports, list)
    assert reports[0]['fiscalDateEnding'] == '2023-03-31'
    assert reports[0]['totalShareholderEquity'] == 200
    assert reports[0]['totalLiabilities'] == 300

def test_compute_ratios():
    income_reports = [{'fiscalDateEnding': '2023-03-31', 'totalRevenue': 500, 'netIncome': 100}]
    balance_reports = [{'fiscalDateEnding': '2023-03-31', 'totalShareholderEquity': 200, 'totalLiabilities': 300}]
    rc = RatioCalc()
    ratios = rc.compute_ratios(income_reports, balance_reports)
    assert abs(ratios[0]['roe'] - 0.5) < 1e-6
    assert abs(ratios[0]['debt_equity'] - 1.5) < 1e-6
    assert abs(ratios[0]['net_profit_margin'] - 0.2) < 1e-6