import pytest
import numpy as np
from src.UI.rdnn import TradingEnv, fetch_ohlcv


def make_dummy_data(n_steps=100):
    o = np.linspace(1, n_steps, n_steps)
    h = o + 1
    l = o - 1
    c = o
    v = np.ones(n_steps)
    return np.stack([o, h, l, c, v], axis=1)


def test_env_reset_and_step():
    data = make_dummy_data()
    env = TradingEnv(data, window_size=5, transaction_cost=0.0, slippage=0.0)
    obs = env.reset()
    assert obs.shape == (5, 5)
    obs, reward, done, info = env.step(2)
    assert isinstance(reward, float)
    assert not done
    assert "net_worth" in info


def test_transaction_cost_and_slippage():
    data = make_dummy_data()
    cost = 0.01
    env = TradingEnv(data, window_size=5, transaction_cost=cost, slippage=cost)
    env.reset()
    _, r1, _, _ = env.step(2)
    _, r2, _, _ = env.step(0)
    assert r1 < 0
    assert r2 < 0


def test_fetch_ohlcv_structure(monkeypatch):
    # Monkeypatch yf.download to return a dummy DataFrame
    import pandas as pd
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Open': np.arange(10),
        'High': np.arange(1, 11),
        'Low': np.arange(-1, 9),
        'Close': np.arange(0, 10),
        'Volume': np.ones(10)
    }, index=dates)
    monkeypatch.setattr('yfinance.download', lambda ticker, period, interval: df)
    arr = fetch_ohlcv('TEST', period='1mo', interval='1d')
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10, 5)

if __name__ == "__main__":
    pytest.main()
