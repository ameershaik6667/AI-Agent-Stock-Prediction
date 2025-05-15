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
from pydantic import RootModel  # For defining JSON output schema
from langchain_openai import ChatOpenAI  # LLM integration

# ----------------------------------------
# CCI Calculation Function
# ----------------------------------------
def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI) for a given DataFrame.
    - Typical Price (TP) = (High + Low + Close) / 3
    - Moving Average (MA) of TP over the window
    - Mean Deviation (MD) of TP from its MA
    - CCI = (TP - MA) / (0.015 * MD)
    """
    # Work on a copy and standardize column names
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Typical price calculation
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    # Rolling mean of TP
    ma = tp.rolling(window=period).mean()
    # Rolling mean deviation of TP
    md = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    # Return the CCI series
    return (tp - ma) / (0.015 * md)

# ----------------------------------------
# CrewAI output model & agent for CCI
# ----------------------------------------
class CCIBuySellAgentOutput(RootModel[dict[str, str]]):
    """RootModel so CrewAI serializes directly to a dict[str, str]."""
    pass

class CCIBuySellAgent(BaseAgent):
    """
    CrewAI agent to generate BUY/SELL/HOLD signals based on CCI values.
    """
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"CCI-based trader for {ticker}",  # Agent role description
            goal="Generate daily BUY/SELL/HOLD signals based on CCI values",
            backstory="You are an expert technical analyst using the Commodity Channel Index.",
            verbose=True,
            tools=[],  # No external tools
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized CCIBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        # Task prompt for the agent
        return Task(
            description=dedent("""
                The global pandas DataFrame `data` has columns:
                  date, high, low, close, cci.

                For each row, output exactly one of: BUY, SELL, or HOLD.
                **Return only** the raw JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD.
                **Do not** include any additional commentary, notes, examples, or placeholders.
            """),
            agent=self,
            output_json=CCIBuySellAgentOutput,
            expected_output="Pure JSON dict mapping dates to BUY/SELL/HOLD, nothing else."
        )

# Initialize a shared LLM for the agent
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# CCI Indicator Wrapper for Backtrader
# ----------------------------------------
class CCIIndicatorBT(bt.Indicator):
    """Backtrader indicator that computes CCI from price data."""
    # Single output line: CCI value
    lines = ('cci',)
    params = (('period', 20),)

    def __init__(self):
        # Ensure enough data before computing
        self.addminperiod(self.p.period)

    def once(self, start, end):
        # Extract OHLC into a DataFrame
        size = self.data.buflen()
        df = pd.DataFrame({
            'high':  [self.data.high[i]  for i in range(size)],
            'low':   [self.data.low[i]   for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
        })
        # Calculate CCI and feed into the indicator line
        cci_series = calculate_cci(df, period=self.p.period)
        for i in range(size):
            self.lines.cci[i] = cci_series.iat[i]

# ----------------------------------------
# CCI Strategy
# ----------------------------------------
class CCIStrategy(bt.Strategy):
    """
    Trading strategy:
    - Buy when CCI < oversold threshold
    - Sell when CCI > overbought threshold
    """
    params = (
        ('period', 20),
        ('oversold', -100),
        ('overbought', 100),
        ('allocation', 1.0),
    )

    def __init__(self):
        self.trade_log = []  # Store trade messages
        # Attach the CCI indicator to the data feed
        self.cci = CCIIndicatorBT(self.data, period=self.p.period)

    def next(self):
        # Called on each new bar
        dt      = self.datas[0].datetime.date(0)
        cci_val = self.cci.cci[0]
        price   = self.data.close[0]

        # Entry signal: oversold
        if not self.position and cci_val < self.p.oversold:
            size = int((self.broker.getcash() * self.p.allocation) // price)
            if size:
                self.buy(size=size)
                msg = f"{dt}: BUY {size} @ {price:.2f} (CCI={cci_val:.2f})"
                self.trade_log.append(msg)
                logging.info(msg)

        # Exit signal: overbought
        elif self.position and cci_val > self.p.overbought:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {price:.2f} (CCI={cci_val:.2f})"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner function
# ----------------------------------------
def run_backtest(strategy_cls, data_feed, cash=10000, commission=0.001):
    """
    Configure Cerebro engine, run the backtest, and extract performance metrics.
    Returns: (summary dict, trade log list, matplotlib figure)
    """
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

    # Gather analyzer results
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
    st.title("CCI Backtest")

    # Sidebar inputs for backtest parameters
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    sd     = st.sidebar.date_input("Start", datetime(2020, 1, 1).date())
    ed     = st.sidebar.date_input("End",   datetime.today().date())
    cash   = st.sidebar.number_input("Cash", 10000)
    comm   = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    period = st.sidebar.number_input("CCI Period", 20, step=1)
    ovb    = st.sidebar.number_input("Overbought Threshold", 100, step=1)
    ovs    = st.sidebar.number_input("Oversold Threshold", -100, step=1)

    if st.sidebar.button("Run Backtest"):
        # Fetch OHLC data
        df = DataFetcher().get_stock_data(
            symbol=ticker, start_date=sd, end_date=ed
        )
        if df.empty:
            st.error("No data fetched—please check ticker and dates.")
            return

        # Calculate CCI and attach to DataFrame
        df['cci'] = calculate_cci(df, period)

        # Generate signals via CrewAI
        globals()['data'] = df.assign(date=df.index)
        agent = CCIBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        crew.kickoff()

        # Display raw JSON signals
        st.subheader("CrewAI Signals (raw JSON)")
        st.code(task.output.json, language="json")

        # Run the Backtrader backtest
        feed  = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(CCIStrategy, feed, cash, comm)

        # Show performance summary and trades
        st.subheader("Performance Summary")
        st.write(perf)
        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        # Plot the equity curve
        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
