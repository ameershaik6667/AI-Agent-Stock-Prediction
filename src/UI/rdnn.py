import gym
from gym import spaces
import numpy as np
import yfinance as yf

class TradingEnv(gym.Env):
    """
    Gym environment for trading based on OHLCV data.
    Observation: window of last T bars (OHLCV).
    Action space: 0=SELL, 1=HOLD, 2=BUY.
    Reward: change in portfolio value minus transaction cost and slippage.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, window_size=10, transaction_cost=0.001, slippage=0.001):
        super(TradingEnv, self).__init__()
        # data: NumPy array of shape (n_steps, 5) representing OHLCV
        self.data = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # define action and observation spaces
        self.action_space = spaces.Discrete(3)
        obs_shape = (window_size, self.data.shape[1])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.reset()

    def reset(self):
        # start at the first point where we have a full window
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.net_worth = 1.0
        return self._get_observation()

    def _get_observation(self):
        return self.data[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        done = False
        price = self.data[self.current_step, 3]  # use close price
        reward = 0.0

        # execute action: SELL (0), HOLD (1), BUY (2)
        if action == 0 and self.position != -1:
            reward -= self._cost(price)
            self.position = -1
            self.entry_price = price
        elif action == 2 and self.position != 1:
            reward -= self._cost(price)
            self.position = 1
            self.entry_price = price

        # calculate P&L and reward
        pnl = self.position * (price - self.entry_price)
        reward += pnl
        self.net_worth += pnl - self._cost(price)

        # advance step
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        info = {"net_worth": self.net_worth}
        return obs, reward, done, info

    def _cost(self, price):
        # simple transaction cost + slippage
        return price * (self.transaction_cost + self.slippage)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Position: {self.position}, Net worth: {self.net_worth:.3f}")


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> np.ndarray:
    """
    Fetch historical OHLCV data for a given ticker using yfinance.
    Returns a NumPy array of shape (n_steps, 5) with columns [Open, High, Low, Close, Volume].
    """
    df = yf.download(ticker, period=period, interval=interval)
    df = df.dropna()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df.values

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    try:
        data_array = fetch_ohlcv(ticker)
        print(f"Fetched {data_array.shape[0]} rows of OHLCV data for {ticker}.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        exit(1)

    window_size = 10
    env = TradingEnv(data_array, window_size=window_size)
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    next_obs, reward, done, info = env.step(1)
    print(f"After one step => reward: {reward:.4f}, net_worth: {info['net_worth']:.4f}")