#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
plt.switch_backend("agg")

import logging
import json
import datetime as dt
from datetime import datetime
from textwrap import dedent
from typing import Dict, List, Tuple

import backtrader as bt
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import crewai
from crewai import Task, Crew, Process

import streamlit as st
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.trix import calculate_trix
from src.Agents.base_agent import BaseAgent

# ──────────────────────────────────────────────────────────
# GLOBALS & DEFAULTS
# ──────────────────────────────────────────────────────────
SYMBOL = "AAPL"
TRIX_DEFAULTS = {
    "ticker":    SYMBOL,
    "length":    14,
    "signal":     9,
    "allocation": 1.0
}
def dict_to_params(d): 
    return tuple((k, v) for k, v in d.items())

# ──────────────────────────────────────────────────────────
# 1) Agent & Output Model
# ──────────────────────────────────────────────────────────
class TrixBuySellAgentOutput(BaseModel):
    output: Dict[str, str]

class TrixBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", llm=None, **kwargs):
        super().__init__(
            role=f"TRIX trader for {ticker}",
            goal="Generate daily BUY/SELL/HOLD signals based on TRIX data",
            backstory="You are an expert TRIX technical analyst.",
            verbose=True,
            tools=[],
            allow_delegation=False,
            llm=llm,
            **kwargs
        )
        self.ticker = ticker

    def buy_sell_decision(self):
        return Task(
            description=dedent(f"""
                The global pandas DataFrame `data` has columns:
                  date, High, Low, Close, TRIX, TRIX_SIGNAL.

                For **every** row in `data`—from the first date to the last—output exactly one of: BUY, SELL, or HOLD.
                Return **only** a JSON object in this shape:
                {{
                  "output": {{
                    "YYYY-MM-DD": "BUY" | "SELL" | "HOLD",
                    … one entry per row of `data` …
                  }}
                }}
                with no additional commentary.
            """),
            agent=self,
            output_json=TrixBuySellAgentOutput,
            expected_output="JSON under key `output` mapping each date in `data` to BUY/SELL/HOLD."
        )

# ──────────────────────────────────────────────────────────
# 2) Crew Wrapper
# ──────────────────────────────────────────────────────────
class TrixCrew:
    def __init__(self, ticker: str, df, length: int, signal: int):
        self.ticker  = ticker
        self.df      = df
        self.length  = length
        self.signal  = signal
        self.llm     = ChatOpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=1500)

    def run(self) -> str:
        trix_df = calculate_trix(self.df.copy(),
                                 length=self.length,
                                 signal=self.signal)
        globals()['data'] = trix_df.assign(date=trix_df.index)
        agent = TrixBuySellAgent(ticker=self.ticker, llm=self.llm)
        task  = agent.buy_sell_decision()
        crew  = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        crew.kickoff()
        return task.output.json

# ──────────────────────────────────────────────────────────
# 3) Backtrader Indicator
# ──────────────────────────────────────────────────────────
class TrixCrewIndicator(bt.Indicator):
    lines = ("signal_num",)
    params = dict_to_params(TRIX_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.length * 3 + self.p.signal)
        df    = self.data._dataname
        raw   = TrixCrew(self.p.ticker, df, self.p.length, self.p.signal).run()
        sigs  = json.loads(raw)["output"]
        mapper = {"BUY": 1, "HOLD": 0, "SELL": -1}
        self.preds = [
            mapper.get(sigs.get(d.strftime("%Y-%m-%d"), "HOLD"), 0)
            for d in df.index
        ]

    def once(self, start, end):
        for i, v in enumerate(self.preds):
            self.lines.signal_num[i] = v

# ──────────────────────────────────────────────────────────
# 4) Strategy
# ──────────────────────────────────────────────────────────
Trade = Tuple[datetime, str, float]

class TrixCrewAIStrategy(bt.Strategy):
    params = dict_to_params(TRIX_DEFAULTS)

    def __init__(self):
        self.trade_log: List[Trade] = []
        self.ind        = TrixCrewIndicator(
            self.data,
            ticker=self.p.ticker,
            length=self.p.length,
            signal=self.p.signal
        )
        self.signal_num = self.ind.signal_num
        self.order      = None
        self.pending    = None

    def bullish_cross(self, prev, curr):
        return prev <= 0 < curr

    def bearish_cross(self, prev, curr):
        return prev >= 0 > curr

    def notify_order(self, order):
        if order.status == order.Completed:
            dt_date = self.data.datetime.date(0)
            kind    = "BUY" if order.isbuy() else "SELL"
            price   = order.executed.price
            self.trade_log.append((dt_date, kind, price))
            if self.pending:
                cash, price = self.broker.getcash(), self.data.close[0]
                size = int((cash/price)*0.95)
                self.order = self.buy(size=size) if self.pending=="LONG" else self.sell(size=size)
                self.pending = None
        self.order = None

    def next(self):
        if self.order:
            return
        prev = self.signal_num[-1] if len(self.signal_num) > 1 else 0
        curr = self.signal_num[0]
        if self.bullish_cross(prev, curr):
            if self.position.size < 0:
                self.order, self.pending = self.close(), "LONG"
            elif not self.position:
                size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                self.order = self.buy(size=size)
        elif self.bearish_cross(prev, curr):
            if self.position.size > 0:
                self.order, self.pending = self.close(), "SHORT"
            elif not self.position:
                size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                self.order = self.sell(size=size)

# ──────────────────────────────────────────────────────────
# 5) Backtest Runner
# ──────────────────────────────────────────────────────────
def run_backtest(strategy, feed, cash, commission):
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(strategy)
    cerebro.adddata(feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="drawdown")

    strat = cerebro.run()[0]
    figs  = cerebro.plot(iplot=False)
    fig   = figs[0][0]

    ax = fig.axes[0]
    for dt_date, kind, price in strat.trade_log:
        marker = "^" if kind == "BUY" else "v"
        color  = "g" if kind == "BUY" else "r"
        ax.scatter(dt_date, price, marker=marker, s=120, color=color)

    sr  = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0.0)
    rtn = strat.analyzers.returns.get_analysis().get("rtot", 0.0) * 100
    dd  = strat.analyzers.drawdown.get_analysis().get("drawdown", 0.0) * 100

    summary = {
        "Sharpe Ratio":     sr,
        "Total Return (%)": rtn,
        "Max Drawdown (%)": dd
    }
    return summary, fig, strat.trade_log

# ──────────────────────────────────────────────────────────
# 6) Streamlit App
# ──────────────────────────────────────────────────────────
def main():
    st.title("TRIX Backtest with CrewAI")
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Ticker", SYMBOL)
    sd     = st.sidebar.date_input("Start", datetime(2020,1,1).date())
    ed     = st.sidebar.date_input("End",   datetime.today().date())
    cash   = st.sidebar.number_input("Cash", 10000)
    comm   = st.sidebar.number_input("Commission", 0.001, step=0.0001)
    length = st.sidebar.number_input("TRIX Length", 14, step=1)
    signal = st.sidebar.number_input("TRIX Signal", 9,  step=1)

    if st.sidebar.button("Run Backtest"):
        df      = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        feed    = bt.feeds.PandasData(dataname=df, fromdate=sd, todate=ed)
        summary, fig, trades = run_backtest(TrixCrewAIStrategy, feed, cash, comm)

        st.subheader("Performance Summary")
        st.json(summary)

        st.subheader("Trade Log")
        for dt_date, kind, price in trades:
            st.write(f"{dt_date} → **{kind}** @ {price:.2f}")

        st.subheader("Equity Curve with Trades")
        st.pyplot(fig)

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main()
