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

# CrewAI + LLM
import crewai
from crewai import Task, Crew, Process
from textwrap import dedent
from langchain_openai import ChatOpenAI

# ----------------------------------------
# Mass Index Calculator
# ----------------------------------------
class MassIndexCalculator:
    def __init__(self, df, ema_period=9, sum_period=25):
        self.df = df.copy()
        self.ema_period = ema_period
        self.sum_period = sum_period

    def calculate(self):
        self.df.columns = [c.lower() for c in self.df.columns]
        if 'date' in self.df.columns:
            self.df.sort_values('date', inplace=True)
        else:
            self.df.sort_index(inplace=True)

        # daily range
        price_range = self.df['high'] - self.df['low']
        # two‑stage EMA
        ema1 = price_range.ewm(span=self.ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.ema_period, adjust=False).mean()
        # ratio and rolling sum
        ratio = ema1 / ema2
        self.df['mass_index'] = ratio.rolling(window=self.sum_period).sum()

        return self.df

# ----------------------------------------
# CrewAI output agent (no Pydantic model)
# ----------------------------------------
class MassIndexBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"Mass Index trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on Mass Index data",
            backstory="You are an expert volatility analyst.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized MassIndexBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        return Task(
            description=dedent(f"""
                The global pandas DataFrame `data` has columns:
                  date, high, low, close, mass_index.

                For each row, output exactly one of: BUY, SELL, or HOLD.
                Return **only** a pure JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD,
                with no additional commentary or notes.
            """),
            agent=self,
            expected_output="Pure JSON dict mapping dates to BUY/SELL/HOLD."
        )

# single shared LLM
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Mass Index Indicator Wrapper for Backtrader
# ----------------------------------------
class MassIndexIndicatorBT(bt.Indicator):
    lines = ('mass_index',)
    params = (
        ('ema_period', 9),
        ('sum_period', 25),
    )

    def __init__(self):
        price_range = self.data.high - self.data.low
        ema1 = bt.indicators.EMA(price_range, period=self.p.ema_period)
        ema2 = bt.indicators.EMA(ema1,        period=self.p.ema_period)
        ratio = ema1 / ema2
        self.lines.mass_index = bt.indicators.SumN(ratio, period=self.p.sum_period)

# ----------------------------------------
# Mass Index Strategy
# ----------------------------------------
class MassIndexStrategy(bt.Strategy):
    params = (
        ('ema_period',      9),
        ('sum_period',     25),
        ('threshold_high', 27.0),
        ('threshold_low',  26.5),
        ('allocation',      1.0),
    )

    def __init__(self):
        self.trade_log = []
        self.mi = MassIndexIndicatorBT(
            self.data,
            ema_period=self.p.ema_period,
            sum_period=self.p.sum_period
        )

    def next(self):
        dt       = self.datas[0].datetime.date(0)
        mi_value = self.mi.mass_index[0]
        close    = self.data.close[0]

        if not self.position and mi_value > self.p.threshold_high:
            size = int((self.broker.getcash() * self.p.allocation) // close)
            self.buy(size=size)
            msg = f"{dt}: BUY  {size} @ {close:.2f}  (MI={mi_value:.2f})"
            self.trade_log.append(msg)
            logging.info(msg)

        elif self.position and mi_value < self.p.threshold_low:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f}  (MI={mi_value:.2f})"
            self.trade_log.append(msg)
            logging.info(msg)

# ----------------------------------------
# Backtest runner
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

    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    s = strat.analyzers.sharpe.get_analysis()

    summary = {
      "Sharpe Ratio":         s.get('sharperatio', 0),
      "Total Return (%)":     r.get('rtot', 0)  * 100,
      "Avg Daily Return (%)": r.get('ravg', 0) * 100,
      "Max Drawdown (%)":     d.get('drawdown', 0) * 100
    }

    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit + CrewAI integration
# ----------------------------------------
def main():
    st.title("Mass Index Backtest")

    st.sidebar.header("Backtest Parameters")
    ticker         = st.sidebar.text_input("Ticker",   "SPY")
    sd             = st.sidebar.date_input("Start",    datetime(2020,1,1).date())
    ed             = st.sidebar.date_input("End",      datetime.today().date())
    cash           = st.sidebar.number_input("Cash",    10000)
    comm           = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    ema_period     = st.sidebar.number_input("EMA Period",  9,  step=1)
    sum_period     = st.sidebar.number_input("Sum Period",  25, step=1)
    threshold_high = st.sidebar.number_input("High Threshold", 27.0, step=0.1)
    threshold_low  = st.sidebar.number_input("Low Threshold",  26.5, step=0.1)

    if st.sidebar.button("Run Backtest"):
        df    = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        mi_df = MassIndexCalculator(df, ema_period=ema_period, sum_period=sum_period).calculate()

        # CrewAI step
        globals()['data'] = mi_df.assign(date=mi_df.index)
        agent = MassIndexBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        signals = crew.kickoff()

        st.subheader("CrewAI Signals (raw JSON)")
        st.code(signals, language="json")

        feed, perf, trades, fig = (
            bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed),
            *run_backtest(MassIndexStrategy, bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed), cash, comm)
        )

        st.subheader("Performance Summary")
        st.write(perf)

        st.subheader("Trade Log")
        for t in trades:
            st.write(t)

        st.subheader("Chart")
        st.pyplot(fig)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()
