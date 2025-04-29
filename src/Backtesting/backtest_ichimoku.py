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
# Ichimoku Cloud Calculator
# ----------------------------------------
class IchimokuCalculator:
    def __init__(self, df,
                 tenkan_period=9, kijun_period=26,
                 senkou_b_period=52, displacement=26,
                 smoothing_factor=1):
        self.df = df.copy()
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.smoothing_factor = smoothing_factor

    def calculate(self):
        self.df.columns = [c.lower() for c in self.df.columns]
        if 'date' in self.df.columns:
            self.df.sort_values('date', inplace=True)
        else:
            self.df.sort_index(inplace=True)

        self.df['tenkan_sen'] = (
            self.df['high'].rolling(self.tenkan_period).max() +
            self.df['low'].rolling(self.tenkan_period).min()
        ) / 2

        self.df['kijun_sen'] = (
            self.df['high'].rolling(self.kijun_period).max() +
            self.df['low'].rolling(self.kijun_period).min()
        ) / 2

        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen'])/2)\
                                   .shift(self.displacement)

        self.df['senkou_span_b'] = (
            self.df['high'].rolling(self.senkou_b_period).max() +
            self.df['low'].rolling(self.senkou_b_period).min()
        )/2
        self.df['senkou_span_b'] = self.df['senkou_span_b'].shift(self.displacement)

        self.df['chikou_span'] = self.df['close'].shift(-self.displacement)

        if self.smoothing_factor > 1:
            for col in ['tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b']:
                self.df[col] = self.df[col].rolling(self.smoothing_factor).mean()

        return self.df

# ----------------------------------------
# CrewAI output model & agent
# ----------------------------------------
class IchimokuBuySellAgentOutput(RootModel[Dict[str, str]]):
    """RootModel so the agent can return a top-level date→signal map."""

class IchimokuBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"Ichimoku Cloud trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on Ichimoku Cloud data",
            backstory="You are an expert Ichimoku Cloud technical analyst.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker
        logging.info(f"Initialized IchimokuBuySellAgent for {ticker}")

    def buy_sell_decision(self):
        return Task(
            description=dedent(f"""
                The global pandas DataFrame `data` has columns:
                  date, high, low, close,
                  tenkan_sen, kijun_sen,
                  senkou_span_a, senkou_span_b,
                  chikou_span.

                For each row, output exactly one of: BUY, SELL, or HOLD.
                Return **only** a pure JSON object mapping YYYY-MM-DD → BUY/SELL/HOLD,
                with no additional commentary or notes.
            """),
            agent=self,
            output_json=IchimokuBuySellAgentOutput,
            expected_output="Pure JSON dict mapping dates to BUY/SELL/HOLD."
        )

# single shared LLM
gpt_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

# ----------------------------------------
# Ichimoku Indicator Wrapper for Backtrader
# ----------------------------------------
class IchimokuIndicatorBT(bt.Indicator):
    lines = ('tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span',)
    params = (
        ('tenkan_period', 9),
        ('kijun_period', 26),
        ('senkou_b_period', 52),
        ('displacement', 26),
        ('smoothing_factor', 1),
    )
    def __init__(self):
        minp = max(self.p.kijun_period, self.p.senkou_b_period) + self.p.displacement
        self.addminperiod(minp)

    def once(self, start, end):
        size = self.data.buflen()
        df = pd.DataFrame({
            'high':  [self.data.high[i]  for i in range(size)],
            'low':   [self.data.low[i]   for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
        })
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')
        res = IchimokuCalculator(df,
            tenkan_period=self.p.tenkan_period,
            kijun_period=self.p.kijun_period,
            senkou_b_period=self.p.senkou_b_period,
            displacement=self.p.displacement,
            smoothing_factor=self.p.smoothing_factor
        ).calculate()
        for i in range(size):
            self.lines.tenkan_sen[i]    = res['tenkan_sen'].iat[i]
            self.lines.kijun_sen[i]     = res['kijun_sen'].iat[i]
            self.lines.senkou_span_a[i] = res['senkou_span_a'].iat[i]
            self.lines.senkou_span_b[i] = res['senkou_span_b'].iat[i]
            self.lines.chikou_span[i]   = res['chikou_span'].iat[i]

# ----------------------------------------
# Ichimoku Strategy driven by CrewAI signals
# ----------------------------------------
class IchimokuStrategy(bt.Strategy):
    params = (
        ('tenkan_period', 9),
        ('kijun_period', 26),
        ('senkou_b_period', 52),
        ('displacement', 26),
        ('smoothing_factor', 1),
        ('allocation', 1.0),
        ('signals', {}),  # dict YYYY-MM-DD → BUY/SELL/HOLD
    )
    def __init__(self):
        self.trade_log = []
        self.signals   = self.p.signals
        self.ichi = IchimokuIndicatorBT(self.data,
            tenkan_period=self.p.tenkan_period,
            kijun_period=self.p.kijun_period,
            senkou_b_period=self.p.senkou_b_period,
            displacement=self.p.displacement,
            smoothing_factor=self.p.smoothing_factor
        )

    def next(self):
        dt_str = self.datas[0].datetime.date(0).strftime('%Y-%m-%d')
        signal = self.signals.get(dt_str, 'HOLD')
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
    st.title("Ichimoku Backtest With CrewAI Signals")

    st.sidebar.header("Backtest Parameters")
    ticker   = st.sidebar.text_input("Ticker","SPY")
    sd       = st.sidebar.date_input("Start", datetime(2020,1,1).date())
    ed       = st.sidebar.date_input("End",   datetime.today().date())
    cash     = st.sidebar.number_input("Cash",       10000)
    comm     = st.sidebar.number_input("Commission",  0.001, step=0.0001)
    tenkan   = st.sidebar.number_input("Tenkan-sen",    9, step=1)
    kijun    = st.sidebar.number_input("Kijun-sen",    26, step=1)
    senkou_b = st.sidebar.number_input("Senkou B",     52, step=1)
    displ    = st.sidebar.number_input("Displacement", 26, step=1)
    smooth   = st.sidebar.number_input("Smoothing",     1, step=1)

    if st.sidebar.button("Run Backtest"):
        # 1) Fetch data
        df = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)

        # 2) Calculate Ichimoku lines
        ich_df = IchimokuCalculator(df,
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_b_period=senkou_b,
            displacement=displ,
            smoothing_factor=smooth
        ).calculate()

        # 3) Get AI signals
        globals()['data'] = ich_df.assign(date=ich_df.index)
        agent = IchimokuBuySellAgent(ticker=ticker, llm=gpt_llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(agents=[agent], tasks=[task], verbose=True, process=Process.sequential)
        crew.kickoff()
        signals = json.loads(task.output.json)

        st.subheader("CrewAI Signals (raw JSON)")
        st.code(task.output.json, language="json")

        # 4) Backtest using AI signals
        feed = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(
            IchimokuStrategy,
            feed,
            cash=cash,
            commission=comm,
            tenkan_period=tenkan,
            kijun_period=kijun,
            senkou_b_period=senkou_b,
            displacement=displ,
            smoothing_factor=smooth,
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
