#!/usr/bin/env python3
# Use non-interactive Matplotlib backend for Streamlit plotting
import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
# Force the Agg backend for matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import streamlit as st  # Streamlit for web UI
from datetime import datetime  # For date operations
import logging  # Logging events
import backtrader as bt  # Backtesting framework
import pandas as pd  # Data manipulation
import numpy as np  # Numeric operations
import sys  # System path modification
from textwrap import dedent  # Clean multi-line strings

# Ensure local project modules import correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher  # Custom data fetcher
from src.Agents.base_agent import BaseAgent  # Base class for CrewAI agents

# CrewAI & Pydantic & LLM imports
import crewai
from crewai import Task, Crew, Process
from pydantic import BaseModel, Extra  # <-- switched here
from langchain_openai import ChatOpenAI  # LLM integration

# ----------------------------------------
# Elder Ray Index Calculation Function
# ----------------------------------------
def calculate_elder_ray(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Calculate the Elder Ray Index (Bull and Bear Power) for a given DataFrame.
    - EMA = exponential moving average of close prices over `period`
    - Bull Power = high - EMA
    - Bear Power = low - EMA
    Returns the DataFrame with added columns: 'ema', 'bull', 'bear'.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df['ema']  = df['close'].ewm(span=period, adjust=False).mean()
    df['bull'] = df['high'] - df['ema']
    df['bear'] = df['low']  - df['ema']
    return df

# ----------------------------------------
# CrewAI output model & agent for Elder Ray
# ----------------------------------------
class ElderRayAgentOutput(BaseModel):
    """
    A catch-all Pydantic model so CrewAI can serialize arbitrary date→signal mappings
    without trying to inspect typed fields.
    """
    class Config:
        extra = Extra.allow

class ElderRayBuySellAgent(BaseAgent):
    """
    CrewAI agent to generate BUY/SELL/HOLD signals based on Elder Ray Index.
    """
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"Elder Ray Index-based trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on Bull and Bear Power",
            backstory="You are an expert technical analyst using the Elder Ray Index.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized ElderRayBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        return Task(
            description=dedent("""\
                The global pandas DataFrame `data` has columns:
                  date, high, low, close, ema, bull, bear.

                For each row, output exactly one of: BUY, SELL, or HOLD.
                **Return only** the raw JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD.
                **Do not** include any additional commentary, notes, examples, or placeholders.
            """),
            agent=self,
            output_json=ElderRayAgentOutput,
            expected_output="Pure JSON dict mapping dates to BUY/SELL/HOLD, nothing else."
        )

# Initialize a shared LLM for the agent
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Elder Ray Indicator Wrapper for Backtrader
# ----------------------------------------
class ElderRayIndicatorBT(bt.Indicator):
    """Backtrader indicator that computes Elder Ray Bull and Bear Power."""
    lines = ('bull', 'bear',)
    params = (('period', 13),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def once(self, start, end):
        size = self.data.buflen()
        df = pd.DataFrame({
            'high':  [self.data.high[i]  for i in range(size)],
            'low':   [self.data.low[i]   for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
        })
        df = calculate_elder_ray(df, period=self.p.period)
        for i in range(size):
            self.lines.bull[i] = df['bull'].iat[i]
            self.lines.bear[i] = df['bear'].iat[i]

# ----------------------------------------
# Elder Ray Strategy
# ----------------------------------------
class ElderRayStrategy(bt.Strategy):
    """
    Trading strategy:
    - Buy when Bull Power > bull_threshold
    - Sell when Bear Power < bear_threshold
    """
    params = (
        ('period', 13),
        ('bull_threshold', 0.5),   # default  0.5
        ('bear_threshold', -0.5),  # default -0.5
        ('allocation', 1.0),
    )

    def __init__(self):
        self.trade_log = []
        self.eri = ElderRayIndicatorBT(self.data, period=self.p.period)

    def next(self):
        dt       = self.datas[0].datetime.date(0)
        bull_val = self.eri.bull[0]
        bear_val = self.eri.bear[0]
        price    = self.data.close[0]

        if not self.position and bull_val > self.p.bull_threshold:
            size = int((self.broker.getcash() * self.p.allocation) // price)
            if size:
                self.buy(size=size)
                msg = f"{dt}: BUY {size} @ {price:.2f} (Bull={bull_val:.2f})"
                self.trade_log.append(msg)
                logging.info(msg)

        elif self.position and bear_val < self.p.bear_threshold:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {price:.2f} (Bear={bear_val:.2f})"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner function
# ----------------------------------------
def run_backtest(strategy_cls, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    logging.info(f"Running {strategy_cls.__name__}…")
    strat = cerebro.run()[0]

    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    summary = {
        "Sharpe Ratio":         strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        "Total Return (%)":     r.get('rtot', 0) * 100,
        "Avg Daily Return (%)": r.get('ravg', 0) * 100,
        "Max Drawdown (%)":     d.get('drawdown', 0) * 100,
    }
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit app with CrewAI integration
# ----------------------------------------
def main():
    st.title("Elder Ray Backtest")

    st.sidebar.header("Backtest Parameters")
    ticker   = st.sidebar.text_input("Ticker", "SPY")
    sd       = st.sidebar.date_input("Start", datetime(2020, 1, 1).date())
    ed       = st.sidebar.date_input("End",   datetime.today().date())
    cash     = st.sidebar.number_input("Cash", 10000)
    comm     = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    period   = st.sidebar.number_input("Elder Ray Period", 13, step=1)
    bull_th  = st.sidebar.number_input("Bull Power Threshold", 0.5,  step=0.1)
    bear_th  = st.sidebar.number_input("Bear Power Threshold", -0.5, step=0.1)

    if st.sidebar.button("Run Backtest"):
        df = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        if df.empty:
            st.error("No data fetched—please check ticker and dates.")
            return

        df = calculate_elder_ray(df, period)

        globals()['data'] = df.assign(date=df.index)
        agent = ElderRayBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        crew.kickoff()

        st.subheader("CrewAI Signals (raw JSON)")
        st.code(task.output.json, language="json")

        feed  = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(ElderRayStrategy, feed, cash, comm)

        st.subheader("Performance Summary")
        st.write(perf)
        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
