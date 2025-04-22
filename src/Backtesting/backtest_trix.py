#!/usr/bin/env python3
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import streamlit as st
from datetime import datetime
import logging
import backtrader as bt
import pandas as pd
import sys
from typing import Dict

# Ensure project modules import correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Agents.base_agent import BaseAgent

# CrewAI + Pydantic + LLM
import crewai
from crewai import Task, Crew, Process
from pydantic import RootModel
from textwrap import dedent
from langchain_openai import ChatOpenAI

# TRIX calculation import
from src.Indicators.trix import calculate_trix

# ----------------------------------------
# TRIX Agent Output model & agent
# ----------------------------------------
class TrixBuySellAgentOutput(RootModel[Dict[str, str]]):
    """RootModel so the agent can return a top-level date→signal map."""

class TrixBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"TRIX trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on TRIX indicator data",
            backstory="You are an expert TRIX technical analyst.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized TrixBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        return Task(
            description=dedent(f"""
                The global pandas DataFrame `data` has columns:
                  date, High, Low, Close, TRIX, TRIX_SIGNAL.

                For each row, output exactly one of: BUY, SELL, or HOLD.
                Return **only** a pure JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD,
                with no additional commentary or notes.
            """),
            agent=self,
            output_json=TrixBuySellAgentOutput,
            expected_output="Pure JSON dict mapping dates to BUY/SELL/HOLD."
        )

# single shared LLM
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# TRIX Indicator Wrapper for Backtrader
# ----------------------------------------
class TrixIndicatorBT(bt.Indicator):
    lines = ('trix','trix_signal',)
    params = (
        ('length', 14),
        ('signal', 9),
    )
    def __init__(self):
        # TRIX requires length * 3 for triple EMA plus signal period
        self.addminperiod(self.p.length * 3 + self.p.signal)

    def once(self, start, end):
        size = self.data.buflen()
        # Build a DataFrame from the historical buffer with correct column names
        df = pd.DataFrame({
            'High':  [self.data.high[i] for i in range(size)],
            'Low':   [self.data.low[i]  for i in range(size)],
            'Close': [self.data.close[i] for i in range(size)],
        })
        # Align dates for JSON mapping (not used by calculate_trix)
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')
        # Calculate TRIX and signal
        res = calculate_trix(df, length=self.p.length, signal=self.p.signal)
        # Assign indicator lines
        for i in range(size):
            self.lines.trix[i]        = res['TRIX'].iat[i]
            self.lines.trix_signal[i] = res['TRIX_SIGNAL'].iat[i]

# ----------------------------------------
# TRIX Strategy
# ----------------------------------------
class TrixStrategy(bt.Strategy):
    params = (
        ('length', 14),
        ('signal', 9),
        ('allocation', 1.0),
    )
    def __init__(self):
        self.trade_log = []
        self.trix_ind = TrixIndicatorBT(self.data,
                                        length=self.p.length,
                                        signal=self.p.signal)
        # Crossover indicator between TRIX and its signal line
        self.crossover = bt.indicators.CrossOver(
            self.trix_ind.trix,
            self.trix_ind.trix_signal
        )

    def next(self):
        dt    = self.datas[0].datetime.date(0)
        close = self.data.close[0]
        # Buy when TRIX crosses above its signal line
        if not self.position and self.crossover > 0:
            size = int((self.broker.getcash() * self.p.allocation) // close)
            self.buy(size=size)
            msg = f"{dt}: BUY {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)
        # Sell when TRIX crosses below its signal line
        elif self.position and self.crossover < 0:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner (unchanged)
# ----------------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')
    logging.info(f"Running {strategy_class.__name__}…")
    strat = cerebro.run()[0]
    r     = strat.analyzers.returns.get_analysis()
    d     = strat.analyzers.drawdown.get_analysis()
    summary = {
      "Sharpe Ratio":          strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
      "Total Return (%)":      r.get('rtot', 0) * 100,
      "Avg Daily Return (%)":  r.get('ravg', 0) * 100,
      "Max Drawdown (%)":      d.get('drawdown', 0) * 100
    }
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit + CrewAI integration
# ----------------------------------------
def main():
    st.title("TRIX Backtest With CrewAI Signals")

    st.sidebar.header("Backtest Parameters")
    ticker     = st.sidebar.text_input("Ticker", "SPY")
    sd         = st.sidebar.date_input("Start", datetime(2020, 1, 1).date())
    ed         = st.sidebar.date_input("End",   datetime.today().date())
    cash       = st.sidebar.number_input("Cash", 10000)
    comm       = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    length     = st.sidebar.number_input("TRIX Length",    14, step=1)
    signal     = st.sidebar.number_input("TRIX Signal",     9, step=1)
    allocation = st.sidebar.number_input("Allocation",      1.0, step=0.01)

    if st.sidebar.button("Run Backtest"):
        # Fetch historical data
        df      = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        # Calculate TRIX
        trix_df = calculate_trix(df.copy(), length=length, signal=signal)

        # CrewAI signals step
        globals()['data'] = trix_df.assign(date=trix_df.index)
        agent = TrixBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        crew.kickoff()

        #st.subheader("CrewAI Signals (raw JSON)")
        #st.code(task.output.json, language="json")

        # Backtest
        feed = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(TrixStrategy, feed, cash, comm)

        st.subheader("Performance Summary")
        st.write(perf)

        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        st.subheader("Chart")
        st.pyplot(fig)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
