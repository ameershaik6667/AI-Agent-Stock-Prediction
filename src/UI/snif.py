import pandas as pd
import numpy as np
import yfinance as yf


class ReturnFetchAgent:
    def __init__(self, source_client=None):
        """
        :param source_client: any object with a .download(...) method
                              default is yfinance
        """
        self.client = source_client or yf

    def fetch_ohlcv(self, tickers, start, end):
        """
        Download OHLCV for given tickers and date range.
        Returns a DataFrame with MultiIndex columns: (Ticker, [Open,High,Low,Close,Volume])
        """
        df = self.client.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
        return df

    def compute_returns(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Given OHLCV with columns like (ticker, field), extract 'Close' and compute
        day-over-day pct_change. Drops the first NaN row.
        Returns: DataFrame of shape (days-1)×(n_tickers)
        """
        if isinstance(ohlcv.columns, pd.MultiIndex):
            close = ohlcv.xs("Close", axis=1, level=1)
        else:
            close = ohlcv["Close"]
        returns = close.pct_change().dropna(how="all")
        return returns

    def clean_and_align(
        self, returns: pd.DataFrame, max_nan_pct: float = 0.5
    ) -> pd.DataFrame:
        """
        1. Drop any dates (rows) where > max_nan_pct of tickers are missing
        2. Forward-fill then back-fill remaining NaNs per ticker
        """
        thresh = int((1 - max_nan_pct) * returns.shape[1])
        returns = returns.dropna(axis=0, thresh=thresh)
        returns = returns.fillna(method="ffill").fillna(method="bfill")
        return returns


if __name__ == "__main__":
    tickers_input = input("Enter tickers (comma-separated, e.g. AAPL,MSFT,GOOG): ")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    agent = ReturnFetchAgent()
    df_ohlcv = agent.fetch_ohlcv(tickers, start_date, end_date)
    print("\nFetched OHLCV data (first 5 rows):")
    print(df_ohlcv.head())

    returns = agent.compute_returns(df_ohlcv)
    print("\nComputed daily returns (first 5 rows):")
    print(returns.head())

    cleaned = agent.clean_and_align(returns)
    print("\nCleaned & aligned returns (first 5 rows):")
    print(cleaned.head())
