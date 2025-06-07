import os
from typing import List, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np
import torch


class PriceLoaderAgent:
    """
    Fetches OHLC data for a universe of tickers and computes daily returns.
    Persists price and returns matrices as NumPy and PyTorch tensors.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        data_dir: str = "data",
    ):
        """
        Args:
            tickers: List of stock symbols to fetch.
            start_date: YYYY-MM-DD format string for the first date.
            end_date: YYYY-MM-DD format string for the last date.
            data_dir: Directory where tensors will be saved.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Will hold DataFrames after fetch
        self.ohlc_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None

    def fetch_ohlc_data(self) -> pd.DataFrame:
        """
        Downloads adjusted close prices for all tickers between start_date and end_date.
        Returns a DataFrame of shape (T, N) where T is # of trading days and
        N is # of tickers. The DataFrame is aligned on trading dates.
        """
        # Use yfinance to download “Adj Close” for all tickers
        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
            threads=True,
        )

        # yfinance returns a multi-indexed DataFrame; extract 'Adj Close'
        if isinstance(raw.columns, pd.MultiIndex):
            adj_close = raw["Adj Close"].copy()
        else:
            # In case only one ticker was passed, columns won't be MultiIndex
            adj_close = raw[["Adj Close"]].rename(columns={"Adj Close": self.tickers[0]})

        # Sort columns by ticker order
        adj_close = adj_close.reindex(columns=self.tickers)

        # Forward-fill or drop NaNs? We'll forward-fill missing prices then drop any remaining NaNs
        adj_close.ffill(inplace=True)
        adj_close.dropna(how="any", inplace=True)

        self.ohlc_df = adj_close
        return adj_close

    def compute_daily_returns(self) -> pd.DataFrame:
        """
        Computes daily simple returns: r_t = (P_t / P_{t-1}) - 1.
        Returns a DataFrame of shape (T-1, N). Index is shifted by one day.
        """
        if self.ohlc_df is None:
            raise RuntimeError("OHLC data not yet fetched. Call fetch_ohlc_data() first.")

        prices = self.ohlc_df
        returns = prices.pct_change().iloc[1:].copy()
        returns.dropna(how="any", inplace=True)

        self.returns_df = returns
        return returns

    def persist_tensors(self) -> Tuple[str, str, str, str]:
        """
        Converts price and returns DataFrames to NumPy and PyTorch tensors and saves them.
        Filenames:
            prices.npy, prices.pt, returns.npy, returns.pt
        Returns:
            Tuple of file paths: (prices_np_path, prices_pt_path, returns_np_path, returns_pt_path)
        """
        if self.ohlc_df is None or self.returns_df is None:
            raise RuntimeError(
                "DataFrames missing. Ensure fetch_ohlc_data() and compute_daily_returns() were called."
            )

        # Convert DataFrames to NumPy arrays
        prices_np = self.ohlc_df.values.astype(np.float32)
        returns_np = self.returns_df.values.astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        prices_pt = torch.from_numpy(prices_np)
        returns_pt = torch.from_numpy(returns_np)

        # Build filenames
        prices_np_path = os.path.join(self.data_dir, "prices.npy")
        prices_pt_path = os.path.join(self.data_dir, "prices.pt")
        returns_np_path = os.path.join(self.data_dir, "returns.npy")
        returns_pt_path = os.path.join(self.data_dir, "returns.pt")

        # Persist to disk
        np.save(prices_np_path, prices_np)
        torch.save(prices_pt, prices_pt_path)
        np.save(returns_np_path, returns_np)
        torch.save(returns_pt, returns_pt_path)

        return prices_np_path, prices_pt_path, returns_np_path, returns_pt_path


if __name__ == "__main__":
    # Prompt the user to enter a list of tickers
    user_input = input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOG): ")
    universe = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]

    # Specify date range (can be adjusted as needed)
    start = "2020-01-01"
    end = "2023-12-31"

    # Instantiate agent
    loader = PriceLoaderAgent(tickers=universe, start_date=start, end_date=end, data_dir="gnn_data")

    # Fetch and align OHLC prices
    prices_df = loader.fetch_ohlc_data()
    print("Price DataFrame head:\n", prices_df.head())

    # Compute daily returns
    returns_df = loader.compute_daily_returns()
    print("\nReturns DataFrame head:\n", returns_df.head())

    # Persist tensors to disk
    p_np, p_pt, r_np, r_pt = loader.persist_tensors()
    print(f"\nSaved:\n - prices (NumPy) at {p_np}\n - prices (PyTorch) at {p_pt}\n - returns (NumPy) at {r_np}\n - returns (PyTorch) at {r_pt}")
