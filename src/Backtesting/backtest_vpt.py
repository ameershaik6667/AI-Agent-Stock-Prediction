#!/usr/bin/env python3
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import streamlit as st
from datetime import datetime
import logging
import backtrader as bt
import pandas as pd
import sys

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.UI.vpt import calculate_vpt

##############################################
# VPT Indicator for Backtrader
##############################################
class VPTIndicatorBT(bt.Indicator):
    lines = ('vpt',)
    params = (
        ('calc_period',      1),
        ('weighting_factor', 1.0),
        ('apply_smoothing',  True),   # now smoothing on by default
        ('smoothing_window', 5),      # 5‑day rolling VPT
    )

    def __init__(self):
        self.addminperiod(self.p.calc_period + self.p.smoothing_window)

    def once(self, start, end):
        size = self.data.buflen()
        df = pd.DataFrame({
            'date':   [self.data.datetime.date(i) for i in range(size)],
            'close':  [self.data.close[i]  for i in range(size)],
            'volume': [self.data.volume[i] for i in range(size)],
        }).set_index('date')

        df = calculate_vpt(
            df.copy(),
            calc_period=self.p.calc_period,
            weighting_factor=self.p.weighting_factor,
            apply_smoothing=self.p.apply_smoothing,
            smoothing_window=self.p.smoothing_window
        )

        for i, (_, row) in enumerate(df.iterrows()):
            self.lines.vpt[i] = row['VPT']

##############################################
# Trend‑Filtered VPT Strategy
##############################################
class TrendFilteredVPTStrategy(bt.Strategy):
    params = (
        ('calc_period',      1),
        ('weighting_factor', 1.0),
        ('apply_smoothing',  True),
        ('smoothing_window', 5),
        ('allocation',       1.0),
        ('sma_period',       200),
    )

    def __init__(self):
        # 200‑day price filter
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close,
                                                     period=self.p.sma_period)

        # VPT momentum
        self.vpt = VPTIndicatorBT(self.data,
                                  calc_period=self.p.calc_period,
                                  weighting_factor=self.p.weighting_factor,
                                  apply_smoothing=self.p.apply_smoothing,
                                  smoothing_window=self.p.smoothing_window)

        self.trade_log = []

    def next(self):
        date  = self.datas[0].datetime.date(0)
        price = self.data.close[0]
        vpt_now, vpt_prev = self.vpt.vpt[0], self.vpt.vpt[-1]
        above_trend = price > self.sma[0]

        if not self.position:
            # enter only if price above its 200‑day SMA & VPT momentum is positive
            if above_trend and vpt_now > vpt_prev:
                cash = self.broker.getcash()
                size = int((cash * self.p.allocation) // price)
                if size:
                    self.buy(size=size)
                    msg = f"{date}: BUY {size} @ {price:.2f}"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            # exit if VPT momentum turns negative or price falls below trend
            if (vpt_now < vpt_prev) or not above_trend:
                size = self.position.size
                self.sell(size=size)
                msg = f"{date}: SELL {size} @ {price:.2f}"
                self.trade_log.append(msg)
                logging.info(msg)

##############################################
# Backtest Runner
##############################################
def run_backtest(data_feed, cash, commission, **kw):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TrendFilteredVPTStrategy, **kw)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    logging.info("Running TrendFilteredVPTStrategy backtest…")
    results = cerebro.run()
    strat   = results[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    rets   = strat.analyzers.returns.get_analysis()
    dd     = strat.analyzers.drawdown.get_analysis()

    perf = {
        "Sharpe Ratio":         sharpe,
        "Total Return":         rets.get('rtot', 0),
        "Avg Daily Return":     rets.get('ravg', 0),
        "Max Drawdown":         dd.get('drawdown', 0),
        "Max Drawdown Duration": dd.get('maxdrawdownperiod', 'N/A'),
    }

    fig = cerebro.plot(iplot=False, show=False)[0][0]
    return perf, strat.trade_log, fig

##############################################
# Streamlit App
##############################################
def main():
    st.title("VPT Backtest")

    st.sidebar.header("Backtest Parameters")
    ticker       = st.sidebar.text_input("Ticker",          value="SPY")
    start         = st.sidebar.date_input("Start Date",    value=datetime(2020,1,1).date())
    end           = st.sidebar.date_input("End Date",      value=datetime.today().date())
    cash         = st.sidebar.number_input("Initial Cash",   value=10000)
    commission   = st.sidebar.number_input("Commission",     value=0.001, step=0.0001)

    st.sidebar.markdown("---")
    st.sidebar.header("VPT Settings")
    cp    = st.sidebar.number_input("Calc Period",       min_value=1,   value=1)
    wf    = st.sidebar.number_input("Weighting Factor",  min_value=0.0, value=1.0, step=0.1)
    sm    = st.sidebar.checkbox("Smooth VPT",         value=True)
    sw    = st.sidebar.number_input("Smoothing Window",  min_value=1, value=5) if sm else 1

    st.sidebar.markdown("---")
    st.sidebar.header("Trend & Strategy")
    sma_p  = st.sidebar.number_input("Trend SMA Period",  min_value=50, value=200)
    alloc  = st.sidebar.slider("Allocation Fraction", 0.0, 1.0, 1.0, step=0.1)

    if st.sidebar.button("Run Backtest"):
        st.write("Fetching data…")
        df   = DataFetcher().get_stock_data(symbol=ticker, start_date=start, end_date=end)
        feed = bt.feeds.PandasData(dataname=df, fromdate=start, todate=end)

        st.write("Running backtest…")
        perf, trades, fig = run_backtest(
            feed, cash, commission,
            calc_period=cp,
            weighting_factor=wf,
            apply_smoothing=sm,
            smoothing_window=sw,
            sma_period=sma_p,
            allocation=alloc
        )

        st.subheader("Performance Summary")
        st.write(f"**Sharpe Ratio:**      {perf['Sharpe Ratio']:.4f}")
        st.write(f"**Total Return:**      {perf['Total Return']*100:.2f}%")
        st.write(f"**Avg Daily Return:**  {perf['Avg Daily Return']*100:.2f}%")
        st.write(f"**Max Drawdown:**      {perf['Max Drawdown']*100:.2f}%")
        st.write(f"**Max Drawdown Duration:** {perf['Max Drawdown Duration']}")

        st.subheader("Trade Log")
        if trades:
            for t in trades:
                st.write(t)
        else:
            st.write("No trades.")

        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
