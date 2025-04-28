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
import json
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

# ----------------------------------------
# Donchian Channels Calculator
# ----------------------------------------
class DonchianCalculator:
    def __init__(self, df, period=20):
        self.df = df.copy()
        self.period = period

    def calculate(self):
        # lowercase columns & sort
        self.df.columns = [c.lower() for c in self.df.columns]
        if 'date' in self.df.columns:
            self.df.sort_values('date', inplace=True)
        else:
            self.df.sort_index(inplace=True)

        # upper channel = highest high over period
        self.df['dc_upper'] = self.df['high'].rolling(self.period).max()
        # lower channel = lowest low over period
        self.df['dc_lower'] = self.df['low'].rolling(self.period).min()
        # optional midline
        self.df['dc_middle'] = (self.df['dc_upper'] + self.df['dc_lower']) / 2

        return self.df

# ----------------------------------------
# CrewAI output model & agent (unchanged)
# ----------------------------------------
class DonchianBuySellAgent(RootModel[Dict[str, str]]):
    """RootModel so the agent can return a top‐level date→signal map."""

class DonchianBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"Donchian Channels trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on Donchian Channels data",
            backstory="You are an expert breakout strategist.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized DonchianBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        return Task(
            description=dedent(f"""
                The global pandas DataFrame `data` has columns:
                  date, high, low, close, donchian_upper, donchian_lower, donchian_mid.

                For each row, output exactly one of: BUY, SELL, or HOLD.
                - BUY if close > donchian_upper
                - SELL if close < donchian_lower
                - HOLD otherwise

                **Output exactly a pure JSON object** mapping YYYY-MM-DD → BUY/SELL/HOLD, no commentary.
            """),
            agent=self,
            output_json=DonchianBuySellAgentOutput,
            expected_output="JSON mapping dates to BUY/SELL/HOLD"
        )

# single shared LLM
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Donchian Indicator Wrapper for Backtrader
# ----------------------------------------
class DonchianIndicatorBT(bt.Indicator):
    lines = ('dc_upper','dc_lower',)
    params = (('period', 20),)

    def __init__(self):
        # need at least `period` bars
        self.addminperiod(self.p.period)

    def once(self, start, end):
        size = self.data.buflen()
        df = pd.DataFrame({
            'high':  [self.data.high[i]  for i in range(size)],
            'low':   [self.data.low[i]   for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
        })
        # synthetic dates for the index
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')
        res = DonchianCalculator(df, period=self.p.period).calculate()

        for i in range(size):
            self.lines.dc_upper[i] = res['dc_upper'].iat[i]
            self.lines.dc_lower[i] = res['dc_lower'].iat[i]

# ----------------------------------------
# Donchian Strategy driven by CrewAI signals
# ----------------------------------------
class DonchianStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('allocation', 1.0),
        ('signals', {}),  # dict YYYY-MM-DD → BUY/SELL/HOLD
    )

    def __init__(self):
        self.trade_log = []
        self.signals   = self.p.signals
        print(f"Signals: {self.signals}")
        self.dc = DonchianIndicatorBT(self.data, period=self.p.period)

    def next(self):
        dt_str = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        signal = self.signals.get(dt_str, 'HOLD')
        #print('signal:', signal)
        price  = self.data.close[0]

        if signal == 'BUY' and not self.position:
            size = int((self.broker.getcash() * self.p.allocation) // price)
            if size:
                self.buy(size=size)
                msg = f"{dt_str}: BUY {size} @ {price:.2f}"
                self.trade_log.append(msg)
                logging.info(msg)

        elif signal == 'SELL' and self.position:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt_str}: SELL {size} @ {price:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)
        # HOLD → do nothing

# ----------------------------------------
# Backtest runner (accepts strategy kwargs)
# ----------------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001, **strategy_kwargs):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **strategy_kwargs)
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
    summary = {
      "Sharpe Ratio":         strat.analyzers.sharpe.get_analysis().get('sharperatio',0),
      "Total Return (%)":     r.get('rtot',0)*100,
      "Avg Daily Return (%)": r.get('ravg',0)*100,
      "Max Drawdown (%)":     d.get('drawdown',0)*100
    }
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit + CrewAI integration
# ----------------------------------------
def main():
    st.title("Donchian Channels Backtest + CrewAI Signals")

    st.sidebar.header("Backtest Parameters")
    ticker   = st.sidebar.text_input("Ticker","SPY")
    sd       = st.sidebar.date_input("Start", datetime(2020,1,1).date())
    ed       = st.sidebar.date_input("End",   datetime.today().date())
    cash     = st.sidebar.number_input("Cash",       10000)
    comm     = st.sidebar.number_input("Commission",  0.001, step=0.0001)
    period   = st.sidebar.number_input("Donchian Period",    20, step=1)

    if st.sidebar.button("Run Backtest"):
        # 1) Fetch data
        df = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)

        # 2) Calculate Donchian Channels
        dc_df = DonchianCalculator(df, period=period).calculate()

        # 3) Get AI signals
        globals()['data'] = dc_df.assign(date=dc_df.index)
        agent = DonchianBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        crew.kickoff()
        signals = json.loads(task.output.json)

        st.subheader("CrewAI Signals (raw JSON)")
        st.code(task.output.json, language="json")

        # 4) Backtest using AI signals
        feed = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(
            DonchianStrategy,
            feed,
            cash=cash,
            commission=comm,
            period=period,
            signals=signals
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
