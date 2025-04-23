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
import re

# Ensure project modules import correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Agents.base_agent import BaseAgent

# CrewAI + LLM
import crewai
from crewai import Task, Crew, Process
from textwrap import dedent
from langchain_openai import ChatOpenAI

gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Donchian Channels Calculator
# ----------------------------------------
class DonchianChannelsCalculator:
    def __init__(self, df, period=20):
        self.df = df.copy()
        self.period = period

    def calculate(self):
        self.df.columns = [c.lower() for c in self.df.columns]
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('date', inplace=True)
        self.df['donchian_upper'] = self.df['high'].rolling(window=self.period).max()
        self.df['donchian_lower'] = self.df['low'].rolling(window=self.period).min()
        # mid‑channel line
        self.df['donchian_mid'] = (self.df['donchian_upper'] + self.df['donchian_lower']) / 2
        return self.df

# ----------------------------------------
# CrewAI Agent for Donchian Signals
# ----------------------------------------
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
            expected_output="JSON mapping dates to BUY/SELL/HOLD"
        )

# ----------------------------------------
# Donchian Indicator for Backtrader
# ----------------------------------------
class DonchianIndicatorBT(bt.Indicator):
    lines = ('upper', 'lower', 'mid',)
    params = (('period', 20),)

    def __init__(self):
        u = bt.indicators.Highest(self.data.high, period=self.p.period)
        l = bt.indicators.Lowest(self.data.low,  period=self.p.period)
        self.lines.upper = u
        self.lines.lower = l
        self.lines.mid = (u + l) / 2

# ----------------------------------------
# Strategy Using CrewAI Signals + Mid‑Channel Rules
# ----------------------------------------
class CrewSignalStrategy(bt.Strategy):
    params = (
        ('signals', {}),
        ('allocation', 1.0),
        ('dc_period', 20),
    )

    def __init__(self):
        # pre‑computed signals from CrewAI
        self.signals = self.p.signals
        # our own Donchian indicator (upper, mid, lower)
        self.dc = DonchianIndicatorBT(self.datas[0], period=self.p.dc_period)
        self.trade_log = []

    def next(self):
        dt    = self.datas[0].datetime.date(0).isoformat()
        action= self.signals.get(dt, 'HOLD')
        close = self.datas[0].close[0]
        up    = self.dc.upper[0]
        mid   = self.dc.mid[0]
        low   = self.dc.lower[0]

        # original breakout logic
        if action == 'BUY' and not self.position:
            size = int((self.broker.getcash() * self.p.allocation) // close)
            if size:
                self.buy(size=size)
                msg = f"{dt}: BUY {size} @ {close:.2f}"
                self.trade_log.append(msg)
                logging.info(msg)

        elif action == 'SELL' and self.position:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f}"
            self.trade_log.append(msg)
            logging.info(msg)

        # mid‑channel entry/exit on HOLD days
        elif action == 'HOLD':
            # enter if price breaks above mid
            if not self.position and close > mid:
                size = int((self.broker.getcash() * self.p.allocation) // close)
                if size:
                    self.buy(size=size)
                    msg = f"{dt}: BUY {size} @ {close:.2f}"
                    self.trade_log.append(msg)
                    logging.info(msg)
            # exit if price falls below mid
            elif self.position and close < mid:
                size = self.position.size
                self.sell(size=size)
                msg = f"{dt}: SELL {size} @ {close:.2f}"
                self.trade_log.append(msg)
                logging.info(msg)

# ----------------------------------------
# Backtest Runner
# ----------------------------------------
def run_backtest(strategy_class, df, signals, cash=10000, commission=0.001, dc_period=20):
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(dataname=df, fromdate=df.index[0], todate=df.index[-1])
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy_class, signals=signals, dc_period=dc_period)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    strat = cerebro.run()[0]
    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    s = strat.analyzers.sharpe.get_analysis()

    summary = {
        'Sharpe Ratio':         s.get('sharperatio', None),
        'Total Return (%)':     r.get('rtot', 0) * 100,
        'Avg Daily Return (%)': r.get('ravg', 0) * 100,
        'Max Drawdown (%)':     d.get('drawdown', 0) * 100,
    }
    # grab the first figure
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit + CrewAI Integration
# ----------------------------------------
def main():
    st.title("Donchian Channels Backtest with CrewAI")
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    sd     = st.sidebar.date_input("Start", datetime(2020,1,1).date())
    ed     = st.sidebar.date_input("End",   datetime.today().date())
    cash   = st.sidebar.number_input("Cash",       10000)
    comm   = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    period = st.sidebar.number_input("Donchian Period", 20, step=1)

    if st.sidebar.button("Run Backtest"):
        # fetch and compute Donchian DataFrame
        df    = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        dc_df = DonchianChannelsCalculator(df, period=period).calculate()
        globals()['data'] = dc_df.assign(date=dc_df.index)

        # get signals from CrewAI
        agent   = DonchianBuySellAgent(ticker=ticker, llm=gpt_llm)
        task    = agent.buy_sell_decision()
        crew    = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        raw_out = crew.kickoff()

        # parse JSON signals
        signals = {}
        if hasattr(raw_out, 'json_dict') and raw_out.json_dict:
            signals = raw_out.json_dict
        else:
            raw_str = None
            if hasattr(raw_out, 'raw') and isinstance(raw_out.raw, str):
                raw_str = raw_out.raw
            elif isinstance(raw_out, dict) and 'raw' in raw_out and isinstance(raw_out['raw'], str):
                raw_str = raw_out['raw']
            elif hasattr(raw_out, 'tasks_output') and raw_out.tasks_output:
                to = raw_out.tasks_output[0]
                if hasattr(to, 'raw') and isinstance(to.raw, str):
                    raw_str = to.raw
                elif isinstance(to, dict) and 'raw' in to and isinstance(to['raw'], str):
                    raw_str = to['raw']
            if raw_str:
                match = re.search(r"\{[\s\S]*\}", raw_str)
                if match:
                    try:
                        signals = json.loads(match.group())
                    except Exception as e:
                        logging.error(f"Failed to parse signals JSON: {e}")
                        signals = {}
                else:
                    logging.error("No JSON object found in raw signals")

        # display and backtest
        st.subheader("CrewAI Signals")
        st.write(signals)

        perf, trades, fig = run_backtest(
            CrewSignalStrategy,
            dc_df,
            signals,
            cash=cash,
            commission=comm,
            dc_period=period
        )

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
