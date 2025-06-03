import yfinance as yf
import sys
import os

class DataFetch:
    """
    Fetch quarterly financial statements for a given ticker via yfinance.
    """
    def __init__(self, num_quarters: int = 10):
        self.num_quarters = num_quarters

    def fetch_income_statement(self, ticker: str) -> list:
        """
        Returns a list of dictionaries—each dict is one quarter's P&L.
        """
        tk = yf.Ticker(ticker)
        inc_df = tk.quarterly_financials.transpose()
        # Sort by fiscal date descending
        inc_df = inc_df.sort_index(ascending=False)
        # Take top num_quarters rows
        top_inc = inc_df.head(self.num_quarters)
        records = []
        for idx, row in top_inc.iterrows():
            records.append({
                "fiscalDateEnding": idx.strftime("%Y-%m-%d"),
                "totalRevenue": row.get("Total Revenue", None),
                "netIncome": row.get("Net Income", None),
            })
        if not records:
            raise ValueError(f"No income statement data found for {ticker}")
        return records

    def fetch_balance_sheet(self, ticker: str) -> list:
        """
        Returns a list of dictionaries—each dict is one quarter's B/S.
        """
        tk = yf.Ticker(ticker)
        bs_df = tk.quarterly_balance_sheet.transpose()
        bs_df = bs_df.sort_index(ascending=False)
        top_bs = bs_df.head(self.num_quarters)
        records = []
        for idx, row in top_bs.iterrows():
            records.append({
                "fiscalDateEnding": idx.strftime("%Y-%m-%d"),
                "totalShareholderEquity": row.get("Total Stockholder Equity", None),
                "totalLiabilities": row.get("Total Liab", None),
            })
        if not records:
            raise ValueError(f"No balance sheet data found for {ticker}")
        return records

class RatioCalc:
    """
    Compute financial ratios (ROE, debt/equity, net profit margin) from raw statements.
    """
    def compute_ratios(self, income_reports: list, balance_reports: list) -> list:
        """
        Given lists of income and balance reports (aligned by quarter), computes:
        - ROE = netIncome / totalShareholderEquity
        - Debt/Equity = totalLiabilities / totalShareholderEquity
        - Net Profit Margin = netIncome / totalRevenue
        Returns a list of dicts with 'fiscalDateEnding' and computed ratios.
        """
        ratios = []
        for inc, bal in zip(income_reports, balance_reports):
            try:
                net_inc = float(inc.get('netIncome', 0) or 0)
                rev = float(inc.get('totalRevenue', 0) or 0)
                equity = float(bal.get('totalShareholderEquity', 0) or 0)
                liabilities = float(bal.get('totalLiabilities', 0) or 0)
            except (TypeError, ValueError):
                net_inc = rev = equity = liabilities = 0.0
            roe = net_inc / equity if equity else None
            debt_equity = liabilities / equity if equity else None
            net_profit_margin = net_inc / rev if rev else None
            ratios.append({
                'fiscalDateEnding': inc.get('fiscalDateEnding'),
                'roe': roe,
                'debt_equity': debt_equity,
                'net_profit_margin': net_profit_margin
            })
        return ratios

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1].strip().upper()
    else:
        ticker = input("Enter stock ticker symbol: ").strip().upper()
    df = DataFetch()
    rc = RatioCalc()

    try:
        income_reports = df.fetch_income_statement(ticker)
        balance_reports = df.fetch_balance_sheet(ticker)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        sys.exit(1)

    ratios = rc.compute_ratios(income_reports, balance_reports)
    print(f"Ratios for {ticker}:")
    for r in ratios:
        print(
            f"Date: {r['fiscalDateEnding']} | ROE: {r['roe']} | "
            f"Debt/Equity: {r['debt_equity']} | Net Profit Margin: {r['net_profit_margin']}"
        )
