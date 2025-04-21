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

# Ensure your project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.UI.gann_main import calculate_gann_hi_lo_activator

##############################################
# Gann Hi‑Lo Activator Indicator for Backtrader
##############################################
class GannHiLoActivator(bt.Indicator):
    lines = ('activator', 'activator_smoothed',)
    params = (('smoothing_period', 10),)

    def __init__(self):
        self.addminperiod(1)

    def once(self, start, end):
        size = self.data.buflen()
        df = pd.DataFrame({
            'high':  [self.data.high[i]  for i in range(size)],
            'low':   [self.data.low[i]   for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
        })
        # Rename to match calculate function expectations
        df.rename(columns={'high':'High', 'low':'Low', 'close':'Close'}, inplace=True)

        gann = calculate_gann_hi_lo_activator(df, smoothing_period=self.p.smoothing_period)

        for i in range(size):
            self.lines.activator[i]          = gann['Gann Hi Lo'].iat[i]
            self.lines.activator_smoothed[i] = gann['Gann Hi Lo Smoothed'].iat[i]

##############################################
# Gann‑based Strategy
##############################################
class GannStrategy(bt.Strategy):
    params = (
        ('smoothing_period', 10),
        ('allocation', 1.0),
    )

    def __init__(self):
        self.trade_log = []
        self.gann = GannHiLoActivator(self.data, smoothing_period=self.p.smoothing_period)

    def next(self):
        today = self.datas[0].datetime.date(0)
        price = float(self.data.close[0])
        act   = float(self.gann.activator_smoothed[0])

        if not self.position:
            if price > act:
                size = int((self.broker.getcash() * self.p.allocation) // price)
                if size:
                    self.buy(size=size)
                    msg = f"{today}: BUY {size} @ {price:.2f}"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            if price < act:
                size = self.position.size
                self.sell(size=size)
                msg = f"{today}: SELL {size} @ {price:.2f}"
                self.trade_log.append(msg)
                logging.info(msg)

##############################################
# Backtest Runner
##############################################
def run_backtest(data_feed, cash, commission, smoothing):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)
    cerebro.addstrategy(GannStrategy, smoothing_period=smoothing)
    cerebro.adddata(data_feed)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,      _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,     _name='drawdown')

    strat = cerebro.run()[0]

    # Performance metrics
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    rtot   = strat.analyzers.returns.get_analysis().get('rtot', 0) * 100
    ravg   = strat.analyzers.returns.get_analysis().get('ravg', 0) * 100

    dd     = strat.analyzers.drawdown.get_analysis()
    maxdd  = dd['max']['drawdown'] * 100
    dur    = dd['max']['len']

    perf = {
        "Sharpe Ratio":            sharpe,
        "Total Return (%)":        rtot,
        "Avg Daily Return (%)":    ravg,
        "Avg Annual Return (%)":   ((1 + ravg/100) ** 252 - 1) * 100,
        "Max Drawdown (%)":        maxdd,
        "Max DD Duration (bars)":  dur,
    }

    fig = cerebro.plot(iplot=False, show=False)[0][0]
    return perf, strat.trade_log, fig

##############################################
# Streamlit UI
##############################################
def main():
    st.title("Gann Hi‑Lo Activator Backtest")

    # Sidebar inputs
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start  = st.sidebar.date_input("Start Date", value=datetime(2020,1,1))
    end    = st.sidebar.date_input("End Date",   value=datetime.today())
    cash   = st.sidebar.number_input("Initial Cash", min_value=1.0, value=10000.0, step=100.0)
    comm   = st.sidebar.number_input("Commission",   min_value=0.0, value=0.001,  step=0.0001)
    smooth = st.sidebar.number_input("Gann Smoothing Period", min_value=1, value=10, step=1)

    if st.sidebar.button("Run Backtest"):
        df   = DataFetcher().get_stock_data(ticker, start, end)
        feed = bt.feeds.PandasData(dataname=df, fromdate=start, todate=end)

        perf, trades, fig = run_backtest(feed, cash, comm, smooth)

        st.subheader("Performance Summary")
        for k, v in perf.items():
            st.write(f"**{k}:** {v:.2f}" if isinstance(v, float) else f"**{k}:** {v}")

        st.subheader("Trade Log")
        if trades:
            for t in trades:
                st.write(t)
        else:
            st.write("No trades executed.")

        st.subheader("Equity Curve")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
