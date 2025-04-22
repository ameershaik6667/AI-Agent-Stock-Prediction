#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------
# CMO Backtest Script with Streamlit + Backtrader + CrewAI
# This script fetches stock data, calculates the Chande Momentum Oscillator (CMO),
# runs a backtest based on CMO signals, and integrates a CrewAI advisory agent.
# ----------------------------------------

import os
# Use a non-interactive backend for matplotlib
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
# Ensure pyplot uses the Agg backend
plt.switch_backend("agg")

import streamlit as st
from datetime import datetime
import logging
import backtrader as bt
import pandas as pd
import sys
from textwrap import dedent

# Add project root to PYTHONPATH so we can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.Data_Retrieval.data_fetcher import DataFetcher

# CrewAI & LLM imports
import crewai
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# ----------------------------------------
# CMO Calculation Class
# ----------------------------------------
class CMOCalculator:
    """
    Computes the Chande Momentum Oscillator (CMO) for a given DataFrame.
    - period: lookback window for momentum calculation
    - calc_method: 'Standard' or 'Absolute'
    - apply_smoothing: None, 'SMA', or 'EMA'
    """
    def __init__(self, df, period=14, calc_method='Standard',
                 apply_smoothing=None, smoothing_period=3, keep_intermediate=False):
        # Copy and lowercase columns for consistency
        self.df = df.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.period = period
        self.calc_method = calc_method
        self.apply_smoothing = apply_smoothing
        self.smoothing_period = smoothing_period
        self.keep_intermediate = keep_intermediate

    def calculate(self):
        # Calculate price change
        self.df['price_change'] = self.df['close'].diff()

        # Compute gains and losses based on the selected method
        if self.calc_method == 'Absolute':
            # Absolute method: treat all moves as gains
            self.df['gain'] = self.df['price_change'].abs()
            self.df['loss'] = 0
        else:
            # Standard: separate positive gains and negative losses
            self.df['gain'] = self.df['price_change'].where(self.df['price_change'] > 0, 0)
            self.df['loss'] = -self.df['price_change'].where(self.df['price_change'] < 0, 0)

        # Rolling sums of gains and losses over the lookback period
        self.df['gain_sum'] = self.df['gain'].rolling(window=self.period).sum()
        self.df['loss_sum'] = self.df['loss'].rolling(window=self.period).sum()

        # Raw CMO calculation
        self.df['cmo'] = 100 * (self.df['gain_sum'] - self.df['loss_sum']) / \
                         (self.df['gain_sum'] + self.df['loss_sum'])

        # Optional smoothing
        if self.apply_smoothing == 'SMA':
            self.df['cmo'] = self.df['cmo'].rolling(window=self.smoothing_period).mean()
        elif self.apply_smoothing == 'EMA':
            self.df['cmo'] = self.df['cmo'].ewm(span=self.smoothing_period, adjust=False).mean()

        # Drop intermediate columns unless debugging
        if not self.keep_intermediate:
            self.df.drop(columns=['price_change', 'gain', 'loss', 'gain_sum', 'loss_sum'], inplace=True)

        return self.df

# ----------------------------------------
# CrewAI Agent for CMO-based Decision
# ----------------------------------------
chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o")

class CMOAnalysisAgent:
    """
    Wraps CrewAI Agent creation and task generation for CMO advice.
    """
    def create_agent(self):
        # Define LLM agent role and capabilities
        return Agent(
            llm=chat_model,
            role="CMO Investment Advisor",
            goal="Provide a BUY, SELL, or HOLD recommendation based on the latest CMO and current stock price.",
            backstory="You are a seasoned technical analyst specializing in momentum indicators.",
            verbose=True,
            tools=[]  # Add any specific tools if desired
        )

    def create_task(self, agent, cmo_df, current_price):
        # Extract the latest CMO value
        latest_cmo = cmo_df['cmo'].iloc[-1]
        # Build the prompt for the agent
        description = dedent(f"""
            Current Stock Price: {current_price:.2f}
            Latest CMO Value:      {latest_cmo:.2f}

            Based on the above, decide if the stock should be BUY, SELL, or HOLD.
            Return **only** the single word BUY, SELL, or HOLD, with a one-sentence rationale.
        """ )
        return Task(
            description=description,
            agent=agent,
            expected_output="BUY, SELL, or HOLD with a one-sentence rationale."
        )

# ----------------------------------------
# Backtrader CMO Indicator Wrapper
# ----------------------------------------
class CMOIndicatorBT(bt.Indicator):
    lines = ('cmo',)
    params = (
        ('period', 14),
        ('calc_method', 'Standard'),
        ('apply_smoothing', None),
        ('smoothing_period', 3),
        ('keep_intermediate', False),
    )

    def __init__(self):
        # Minimum lookback bars (including smoothing)
        extra = self.p.smoothing_period if self.p.apply_smoothing else 0
        self.addminperiod(self.p.period + extra)

    def once(self, start, end):
        # Build a small DataFrame of closing prices
        size = self.data.buflen()
        df = pd.DataFrame({'close': [self.data.close[i] for i in range(size)]})
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')
        # Reuse CMOCalculator
        res = CMOCalculator(
            df,
            period=self.p.period,
            calc_method=self.p.calc_method,
            apply_smoothing=self.p.apply_smoothing,
            smoothing_period=self.p.smoothing_period,
            keep_intermediate=self.p.keep_intermediate
        ).calculate()
        # Populate indicator line
        for i in range(size):
            self.lines.cmo[i] = res['cmo'].iat[i]

# ----------------------------------------
# Simple CMO Strategy
# ----------------------------------------
class CMOStrategy(bt.Strategy):
    params = (
        ('period', 14),
        ('threshold', 50),
        ('calc_method', 'Standard'),
        ('apply_smoothing', None),
        ('smoothing_period', 3),
        ('allocation', 1.0),
    )

    def __init__(self):
        # Store executed trades
        self.trade_log = []
        # Attach the CMO indicator to our data
        self.cmo = CMOIndicatorBT(
            self.data,
            period=self.p.period,
            calc_method=self.p.calc_method,
            apply_smoothing=self.p.apply_smoothing,
            smoothing_period=self.p.smoothing_period
        )
        self.last_signal = None

    def next(self):
        # Current date, CMO value, and price
        dt = self.datas[0].datetime.date(0)
        cmo_val = self.cmo.cmo[0]
        close = self.data.close[0]
        th = self.p.threshold

        # Buy when CMO crosses above +threshold
        if not self.position and self.last_signal != 'BUY' and cmo_val > th:
            size = int((self.broker.getcash() * self.p.allocation) // close)
            if size:
                self.buy(size=size)
                msg = f"{dt}: BUY {size} @ {close:.2f} (CMO={cmo_val:.1f})"
                self.trade_log.append(msg)
                logging.info(msg)
                self.last_signal = 'BUY'

        # Sell when CMO crosses below -threshold
        elif self.position and self.last_signal != 'SELL' and cmo_val < -th:
            size = self.position.size
            self.sell(size=size)
            msg = f"{dt}: SELL {size} @ {close:.2f} (CMO={cmo_val:.1f})"
            self.trade_log.append(msg)
            logging.info(msg)
            self.last_signal = 'SELL'

# ----------------------------------------
# Backtest runner using Backtrader analyzers
# ----------------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    # Add performance analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    logging.info(f"Running {strategy_class.__name__}…")
    strat = cerebro.run()[0]
    # Extract analyzer results
    r = strat.analyzers.returns.get_analysis()
    d = strat.analyzers.drawdown.get_analysis()
    summary = {
        "Sharpe Ratio": strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
        "Total Return (%)": r.get('rtot', 0) * 100,
        "Avg Daily Return (%)": r.get('ravg', 0) * 100,
        "Max Drawdown (%)": d.get('drawdown', 0) * 100
    }
    # Generate equity curve plot
    fig = cerebro.plot(iplot=False)[0][0]
    return summary, strat.trade_log, fig

# ----------------------------------------
# Streamlit App Entry Point
# ----------------------------------------
def main():
    st.title("CMO Backtest")

    # Sidebar inputs for backtest and CMO parameters
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", "SPY")
    sd     = st.sidebar.date_input("Start", datetime(2020,1,1).date())
    ed     = st.sidebar.date_input("End",   datetime.today().date())
    cash   = st.sidebar.number_input("Cash",    10000)
    comm   = st.sidebar.number_input("Commission", 0.001, step=0.0001)

    st.sidebar.subheader("CMO Settings")
    period = st.sidebar.number_input("CMO Period", min_value=1, value=14)
    method = st.sidebar.selectbox("Method", ["Standard", "Absolute"])
    smooth = st.sidebar.selectbox("Smoothing", [None, "SMA", "EMA"], format_func=lambda x: "None" if x is None else x)
    sp     = st.sidebar.number_input("Smooth Period", min_value=1, value=3)
    keep   = st.sidebar.checkbox("Keep Intermediate Columns", value=False)
    thresh = st.sidebar.slider("Threshold", min_value=0, max_value=100, value=50)

    # Run logic on button click
    if st.sidebar.button("Run Backtest"):
        # 1) Fetch data
        raw_df = DataFetcher().get_stock_data(symbol=ticker, start_date=sd, end_date=ed)
        raw_df.columns = [c.lower() for c in raw_df.columns]
        raw_df = raw_df.assign(date=raw_df.index)

        # 2) Calculate CMO
        cmo_df = CMOCalculator(raw_df,
                  period=period,
                  calc_method=method,
                  apply_smoothing=smooth,
                  smoothing_period=sp,
                  keep_intermediate=keep
                ).calculate()
        cmo_df = cmo_df.assign(date=cmo_df.index)

        # Display calculated CMO
        st.subheader("Calculated CMO (last 20 rows)")
        st.dataframe(cmo_df[['date','close','cmo']].tail(20))

        # 3) CrewAI advisory
        current_price = cmo_df['close'].iloc[-1]
        advisor       = CMOAnalysisAgent()
        agent         = advisor.create_agent()
        task          = advisor.create_task(agent, cmo_df, current_price)
        crew          = Crew(agents=[agent], tasks=[task], verbose=True)
        decision      = crew.kickoff()

        # 4) Execute backtest
        feed = bt.feeds.PandasData(dataname=raw_df, fromdate=sd, todate=ed)
        perf, trades, fig = run_backtest(CMOStrategy, feed, cash=cash, commission=comm)

        # Show results
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
