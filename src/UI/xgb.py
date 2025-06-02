import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

# === Data Ingestion & Windowing ===

def fetch_ohlcv(tickers: List[str], start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    """
    Fetch OHLCV data for given tickers from yfinance.
    Returns a DataFrame with a MultiIndex [Ticker, Date].
    """
    data = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=False)
    df_list = []
    for t in tickers:
        df_t = data[t].copy()
        df_t['Ticker'] = t
        df_list.append(df_t)
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    df.set_index(['Ticker', 'Date'], inplace=True)
    return df

def preprocess_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Impute or drop missing data. 
    method: 'ffill', 'bfill', or 'drop'
    """
    if method in ['ffill', 'bfill']:
        df = df.groupby(level=0).apply(lambda x: x.fillna(method=method))
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError("method must be 'ffill', 'bfill', or 'drop'")
    df = df.dropna()
    return df

def normalize_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Z-score normalization for each ticker separately over the specified feature columns.
    """
    scaler = StandardScaler()
    df_norm = df.copy()
    for t in df.index.get_level_values(0).unique():
        idx = df.index.get_level_values(0) == t
        df_norm.loc[idx, features] = scaler.fit_transform(df.loc[idx, features])
    return df_norm

def slice_windows(
    df: pd.DataFrame,
    features: List[str],
    window_size: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice DataFrame into overlapping windows of length `window_size`.
    Returns:
      X: numpy array of shape (num_windows, window_size, num_features)
      y: numpy array of next-day returns (num_windows,)
    """
    X, y = [], []
    for t in df.index.get_level_values(0).unique():
        df_t = df.loc[t]
        values = df_t[features].values
        closes = df_t['Close'].values
        for start in range(0, len(df_t) - window_size, stride):
            end = start + window_size
            X.append(values[start:end])
            ret = (closes[end] - closes[end - 1]) / closes[end - 1]
            y.append(ret)
    return np.array(X), np.array(y)

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split arrays X and y into train/val/test sets according to the given ratios.
    """
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_idx = idx[: n - test_size - val_size]
    val_idx = idx[n - test_size - val_size : n - test_size]
    test_idx = idx[n - test_size :]
    return {
        'train': (X[train_idx], y[train_idx]),
        'val': (X[val_idx], y[val_idx]),
        'test': (X[test_idx], y[test_idx])
    }

# === Example Usage ===
if __name__ == '__main__':
    # Prompt the user to enter a stock ticker symbol
    ticker_input = input("Enter a stock ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker_input:
        print("No ticker provided. Exiting.")
        exit(1)

    # Prompt for start and end dates
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    try:
        # Validate basic format
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
    except Exception:
        print("Invalid date format. Please use YYYY-MM-DD. Exiting.")
        exit(1)

    tickers = [ticker_input]
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    window_size = 20

    # 1. Fetch OHLCV data
    df = fetch_ohlcv(tickers, start=start_date, end=end_date)

    # 2. Impute missing values
    df = preprocess_data(df, method='ffill')

    # 3. Normalize features
    df = normalize_features(df, features)

    # 4. Slice into windows and generate labels
    X, y = slice_windows(df, features, window_size)

    # 5. Split into train/val/test
    splits = train_val_test_split(X, y, val_ratio=0.2, test_ratio=0.1, shuffle=True)

    print("\nShapes:")
    print("  Train X:", splits['train'][0].shape, " Train y:", splits['train'][1].shape)
    print("  Val   X:", splits['val'][0].shape,   " Val   y:", splits['val'][1].shape)
    print("  Test  X:", splits['test'][0].shape,  " Test  y:", splits['test'][1].shape)
