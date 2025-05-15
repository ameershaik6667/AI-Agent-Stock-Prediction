#!/usr/bin/env python3
# ATR Backtest Script with Streamlit, Backtrader, and CrewAI

import os
# Use non-interactive backend for matplotlib (needed for Streamlit)
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
# Force the Agg backend for matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("agg")  # ensure Agg is active

import streamlit as st  # Streamlit for web UI
from datetime import datetime  # for date operations
import logging  # for logging events

import backtrader as bt  # core backtesting framework
import backtrader.indicators as btind  # built-in indicators for Backtrader
import pandas as pd  # data manipulation
import sys  # to modify sys.path
from textwrap import dedent  # for clean multi-line strings

# Add project root so we can import custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.Data_Retrieval.data_fetcher import DataFetcher  # custom data fetcher
from src.Agents.base_agent import BaseAgent  # Base class for CrewAI agents

import crewai  # CrewAI framework for agent-based analysis
from crewai import Task, Crew, Process  # Task definition, Crew orchestration
from langchain_openai import ChatOpenAI  # LLM interface

# ----------------------------------------
# ATR Calculator (for DataFrame → ATR)
# ----------------------------------------
class ATRCalculator:
    """
    Computes the Average True Range (ATR) indicator columns
    from a raw OHLC DataFrame.
    """
    def __init__(self, df, period=14):
        # Copy input DataFrame and set ATR period
        self.df = df.copy()
        self.period = period

    def calculate(self):
        # Normalize column names to lowercase
        self.df.columns = [c.lower() for c in self.df.columns]
        # Sort by date if present, else by index
        if 'date' in self.df.columns:
            self.df.sort_values('date', inplace=True)
        else:
            self.df.sort_index(inplace=True)

        # Calculate True Range components:
        # high_low: difference between high and low
        self.df['high_low']   = self.df['high'] - self.df['low']
        # high_close: abs(high - previous close)
        self.df['high_close'] = (self.df['high'] - self.df['close'].shift(1)).abs()
        # low_close: abs(low - previous close)
        self.df['low_close']  = (self.df['low']  - self.df['close'].shift(1)).abs()
        # True Range is the max of the three
        self.df['true_range'] = self.df[['high_low','high_close','low_close']].max(axis=1)
        # ATR is the rolling mean of True Range over 'period'
        self.df['atr']        = self.df['true_range'].rolling(self.period).mean()

        return self.df

# ----------------------------------------
# ATR Buy/Sell Agent (no Pydantic model)
# ----------------------------------------
class ATRBuySellAgent(BaseAgent):
    """
    CrewAI agent that produces BUY/SELL/HOLD signals
    based on the ATR column in a DataFrame.
    """
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"ATR trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on ATR data",
            backstory="You are an expert technical analyst using the ATR.",
            verbose=True,
            tools=[],  # no external tools
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized ATRBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        # Task prompt describing DataFrame columns and output format
        return Task(
            description=dedent("""
                The global pandas DataFrame `data` has columns:
                  date, high, low, close, atr.
                For each row, output exactly one of: BUY, SELL, or HOLD.
                Return **only** a pure JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD,
                with no extra commentary.
            """),
            agent=self,
            expected_output="Pure JSON dict mapping dates to BUY/SELL/HOLD."
        )

# Shared LLM instance for all CrewAI agents
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# ATR Indicator wrapper for Backtrader
# ----------------------------------------
class ATRIndicatorBT(bt.Indicator):
    """
    Wraps Backtrader's built-in ATR indicator into a custom Indicator.
    """
    lines = ('atr',)
    params = (('period', 14),)

    def __init__(self):
        # Use built-in ATR and expose it on self.lines.atr
        self.atr = btind.ATR(self.data, period=self.p.period)
        self.lines.atr = self.atr

# ----------------------------------------
# ATR Breakout Strategy
# ----------------------------------------
class ATRStrategy(bt.Strategy):
    """
    Simple ATR breakout strategy:
      - Go long when price > previous_close + multiplier*ATR
      - Exit long when price < previous_close - multiplier*ATR
      - (Optionally go short / cover symmetric)
    """
    params = (
        ('atr_period',14),     # lookback for ATR
        ('multiplier',1.0),    # breakout threshold
        ('allocation',1.0),    # fraction of cash per trade
    )

    def __init__(self):
        # Store a human-readable trade log
        self.trade_log = []
        # Attach our ATR line
        self.atr = ATRIndicatorBT(self.data, period=self.p.atr_period).atr

    def next(self):
        # Called on each new bar
        dt         = self.datas[0].datetime.date(0)
        close      = self.data.close[0]
        prev_close = self.data.close[-1]
        atr_value  = self.atr[0] or 0.0

        # Define entry and exit levels
        entry = prev_close + self.p.multiplier * atr_value
        exit_ = prev_close - self.p.multiplier * atr_value

        # If no open position and price breaks above entry, BUY
        if not self.position and close > entry:
            size = int((self.broker.getcash()*self.p.allocation)//close)
            self.buy(size=size)
            msg = f"{dt}: BUY {size} @ {close:.2f} (> {entry:.2f})"
            self.trade_log.append(msg)
            logging.info(msg)

        # If long position and price drops below exit, SELL
        elif self.position and close < exit_:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f} (< {exit_:.2f})"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner function
# ----------------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    """
    Sets up Cerebro, adds strategy, data, and analyzers, runs the backtest,
    and returns performance summary, trade log, and a matplotlib Figure.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add analyzers for Sharpe, Returns, DrawDown
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    logging.info(f"Running {strategy_class.__name__}…")
    strat = cerebro.run()[0]  # execute and get the strategy instance

    # Extract analyzer results
    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    summary = {
      "Sharpe Ratio":         strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
      "Total Return (%)":     r.get('rtot', 0)*100,
      "Avg Daily Return (%)": r.get('ravg', 0)*100,
      "Max Drawdown (%)":     d.get('drawdown', 0)*100,
    }

    # Generate plot and grab the Figure
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit + CrewAI integration
# ----------------------------------------
def main():
    # App title
    st.title("ATR Backtest with CrewAI Signals")

    # Sidebar inputs
    st.sidebar.header("Parameters")
    ticker     = st.sidebar.text_input("Ticker", "SPY")
    sd         = st.sidebar.date_input("Start", datetime(2020,1,1).date())
    ed         = st.sidebar.date_input("End",   datetime.today().date())
    cash       = st.sidebar.number_input("Cash", 10000)
    comm       = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    atr_period = st.sidebar.number_input("ATR Period", 14, step=1)
    multiplier = st.sidebar.number_input("Multiplier", 1.0, step=0.1)
    allocation = st.sidebar.slider("Allocation", 0.1, 1.0, 1.0, step=0.1)

    # Run logic when button clicked
    if st.sidebar.button("Run Backtest"):
        # 1) Fetch data via custom DataFetcher
        df     = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        # 2) Compute ATR columns
        atr_df = ATRCalculator(df, period=atr_period).calculate()

        # 3) CrewAI trading signals
        globals()['data'] = atr_df.assign(date=atr_df.index)
        agent            = ATRBuySellAgent(ticker=ticker, llm=gpt_llm)
        task             = agent.buy_sell_decision()
        crew             = Crew(
                              agents=[agent],
                              tasks=[task],
                              verbose=True,
                              process=Process.sequential
                          )
        outputs = crew.kickoff()

        # Robust extraction of the raw JSON result
        first = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if hasattr(first, 'raw'):
            raw = first.raw
        elif hasattr(first, 'result'):
            raw = first.result
        elif hasattr(first, 'content'):
            raw = first.content
        elif callable(getattr(first, 'json', None)):
            raw = first.json()
        else:
            raw = str(first)

        # Display CrewAI output
        st.subheader("CrewAI Signals (raw JSON)")
        st.code(raw, language="json")

        # 4) Backtest with Backtrader
        feed, = [bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)]
        perf, trades, fig = run_backtest(ATRStrategy, feed, cash, comm)

        # Show performance table
        st.subheader("Performance Summary")
        st.write(perf)

        # Show trade-by-trade log
        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        # Plot the P/L curve
        st.subheader("Chart")
        st.pyplot(fig)

if __name__ == '__main__':
    # Configure root logger
    logging.basicConfig(level=logging.INFO)
    main()
