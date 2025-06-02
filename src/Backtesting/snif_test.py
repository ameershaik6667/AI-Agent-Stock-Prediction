import pandas as pd
import numpy as np
import pytest

from src.UI.snif import ReturnFetchAgent


class DummyClient:
    """
    Simulates yfinance-like download(...) for testing.
    Builds a DataFrame with specified dates, tickers, and Close prices.
    """
    def __init__(self, close_data: pd.DataFrame):
        """
        :param close_data: DataFrame of shape (n_dates × n_tickers), indices=dates, columns=tickers
                           Represents Close prices for each ticker on each date.
        """
        self.close_data = close_data

    def download(self, tickers, start, end, group_by, auto_adjust, progress):
        dates = self.close_data.index
        # Build a MultiIndex DataFrame with columns (ticker, [Open,High,Low,Close,Volume]).
        # For simplicity, set Open=High=Low=Close and Volume=1 for each ticker.
        arrays = []
        data = []
        for ticker in tickers:
            price_series = self.close_data[ticker]
            arrays.append((ticker, "Open"))
            arrays.append((ticker, "High"))
            arrays.append((ticker, "Low"))
            arrays.append((ticker, "Close"))
            arrays.append((ticker, "Volume"))

            # For each date, replicate the Close price into O/H/L, and set Volume=1
            for dt in dates:
                # O, H, L = Close; Volume = 1
                pass

        # Construct the DataFrame row by row
        multi_cols = pd.MultiIndex.from_tuples(arrays)
        full_data = np.zeros((len(dates), len(multi_cols)), dtype=float)

        for col_idx, (ticker, field) in enumerate(multi_cols):
            if field == "Volume":
                full_data[:, col_idx] = 1.0
            else:
                full_data[:, col_idx] = self.close_data[ticker].values

        return pd.DataFrame(full_data, index=dates, columns=multi_cols)


@pytest.fixture
def simple_close_df():
    """
    Creates a DataFrame of Close prices for 3 tickers over 5 dates,
    with some missing values to test data integrity logic.
    """
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    data = {
        "AAA": [10.0,  np.nan, 12.0, 13.0, np.nan],
        "BBB": [np.nan, 20.0,  np.nan, 22.0, 23.0],
        "CCC": [30.0, 31.0,  np.nan, np.nan, 33.0],
    }
    return pd.DataFrame(data, index=dates)


def test_compute_returns_drops_first_nan(simple_close_df):
    """
    Verify that compute_returns drops the first NaN row and calculates pct_change.
    """
    agent = ReturnFetchAgent(source_client=DummyClient(simple_close_df))
    ohlcv = agent.fetch_ohlcv(["AAA", "BBB", "CCC"], "2025-01-01", "2025-01-06")
    returns = agent.compute_returns(ohlcv)

    # Manually compute expected pct_change on "Close" columns
    close = simple_close_df
    expected = close.pct_change().dropna(how="all")

    # After pct_change, the first row should be dropped
    pd.testing.assert_frame_equal(
        returns, expected,
        check_names=False,
        obj="compute_returns output mismatch"
    )


def test_clean_and_align_no_nans(simple_close_df):
    """
    After compute_returns and clean_and_align, there should be no NaNs left.
    Also, rows with >50% missing values should be dropped.
    """
    agent = ReturnFetchAgent(source_client=DummyClient(simple_close_df))
    ohlcv = agent.fetch_ohlcv(["AAA", "BBB", "CCC"], "2025-01-01", "2025-01-06")
    returns = agent.compute_returns(ohlcv)

    # Introduce NaNs into returns to simulate missing data
    # (pct_change will already have some NaNs where original data was NaN)
    cleaned = agent.clean_and_align(returns, max_nan_pct=0.5)

    # 1) No NaNs remain
    assert not cleaned.isna().any().any(), "clean_and_align left NaNs behind"

    # 2) Check that rows with >50% missing were dropped.
    # Original pct_change on simple_close_df:
    # pct_change before dropna:
    #            AAA       BBB       CCC
    # 2025-01-01    NaN       NaN       NaN
    # 2025-01-02    NaN  0.000000  0.033333
    # 2025-01-03  0.200       NaN       NaN
    # 2025-01-04  0.083  0.100000       NaN
    # 2025-01-05    NaN  0.045455  0.015152
    #
    # Rows with missing counts (>50%):
    # 2025-01-01: 3/3 missing → dropped by dropna(how="all") in compute_returns
    # 2025-01-02: 1/3 missing → keep (1/3 < 0.5)
    # 2025-01-03: 2/3 missing → drop (2/3 > 0.5)
    # 2025-01-04: 1/3 missing → keep
    # 2025-01-05: 1/3 missing → keep
    #
    expected_indices = pd.to_datetime(["2025-01-02", "2025-01-04", "2025-01-05"])
    assert list(cleaned.index) == list(expected_indices), "Rows dropped incorrectly"


def test_full_pipeline_shape_and_nan_free(simple_close_df):
    """
    End-to-end pipeline: fetch → compute_returns → clean_and_align.
    Verify final shape and NaN-free guarantee.
    """
    agent = ReturnFetchAgent(source_client=DummyClient(simple_close_df))
    ohlcv = agent.fetch_ohlcv(["AAA", "BBB", "CCC"], "2025-01-01", "2025-01-06")
    returns = agent.compute_returns(ohlcv)
    cleaned = agent.clean_and_align(returns, max_nan_pct=0.5)

    # Expect 3 rows (dates with ≤50% missing) and 3 tickers
    assert cleaned.shape == (3, 3), f"Unexpected cleaned shape: {cleaned.shape}"
    assert not cleaned.isna().any().any(), "Final cleaned DataFrame still contains NaNs"


def test_edge_case_all_missing_ticker():
    """
    If one ticker has all NaN closes, compute_returns should drop that column before or
    after cleaning, leaving only valid tickers.
    """
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    data = {
        "ALLNAN": [np.nan, np.nan, np.nan, np.nan],
        "GOOD":   [10.0, 11.0, 12.0, 13.0],
    }
    close_df = pd.DataFrame(data, index=dates)
    agent = ReturnFetchAgent(source_client=DummyClient(close_df))
    ohlcv = agent.fetch_ohlcv(["ALLNAN", "GOOD"], "2025-01-01", "2025-01-05")
    returns = agent.compute_returns(ohlcv)

    # After pct_change, "ALLNAN" column should remain all NaNs; "GOOD" has valid pct_changes
    assert "ALLNAN" in returns.columns and returns["ALLNAN"].isna().all()
    assert "GOOD" in returns.columns and not returns["GOOD"].isna().any()

    cleaned = agent.clean_and_align(returns, max_nan_pct=0.5)

    # The row drop logic: rows where >50% tickers missing.
    # For "ALLNAN" & "GOOD":
    # pct_change matrix (before dropna):
    #            ALLNAN      GOOD
    # 2025-01-01    NaN       NaN
    # 2025-01-02    NaN  0.100000
    # 2025-01-03    NaN  0.090909
    # 2025-01-04    NaN  0.083333
    # After dropna(how="all"), first row is removed
    # Remaining rows have only 1/2 columns missing (0.5), so they should be kept
    # Imputation will fill ALLNAN with GOOD's previous values
    assert not cleaned.isna().any().any(), "ALLNAN ticker should be filled via backfill/ffill"
    assert "ALLNAN" in cleaned.columns and "GOOD" in cleaned.columns
