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

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.UI.elder_ray_index import calculate_elder_ray_index

##############################################
# Elder‑Ray Indicator Wrapper for Backtrader
##############################################
class ElderRayIndicatorBT(bt.Indicator):
    """
    Backtrader indicator for the Elder‑Ray Index:
      - ma (moving average)
      - bull_power (high – ma)
      - bear_power (low – ma)
    """
    lines = ('ma', 'bull_power', 'bear_power',)
    params = (
        ('ma_period', 13),
        ('ma_type', 'EMA'),      # 'EMA' or 'SMA'
        ('price_column', 'close'),
    )

    def __init__(self):
        # need at least ma_period bars to compute the MA
        self.addminperiod(self.p.ma_period)

    def once(self, start, end):
        size = self.data.buflen()
        # build a DataFrame from the feed
        df = pd.DataFrame({
            'high':   [self.data.high[i]  for i in range(size)],
            'low':    [self.data.low[i]   for i in range(size)],
            'open':   [self.data.open[i]  for i in range(size)],
            'close':  [self.data.close[i] for i in range(size)],
        })
        # lowercase columns to satisfy calculate_elder_ray_index
        df.columns = [c.lower() for c in df.columns]

        # calculate Elder‑Ray
        result = calculate_elder_ray_index(
            df,
            ma_period=self.p.ma_period,
            ma_type=self.p.ma_type,
            price_column=self.p.price_column
        )

        # assign to lines
        for i in range(size):
            self.lines.ma[i]          = result['ma'].iloc[i]
            self.lines.bull_power[i]  = result['bull power'].iloc[i]
            self.lines.bear_power[i]  = result['bear power'].iloc[i]

##############################################
# Elder‑Ray Strategy
##############################################
class ElderRayStrategy(bt.Strategy):
    """
    Entry:  go long when Bull Power > 0
    Exit:   close when Bear Power < 0
    """
    params = (
        ('ma_period', 13),
        ('ma_type', 'EMA'),
        ('price_column', 'close'),
        ('allocation', 1.0),
    )

    def __init__(self):
        self.trade_log = []
        self.eri = ElderRayIndicatorBT(
            self.data,
            ma_period=self.p.ma_period,
            ma_type=self.p.ma_type,
            price_column=self.p.price_column
        )

    def next(self):
        dt    = self.datas[0].datetime.date(0)
        price = self.data.close[0]
        bull  = self.eri.bull_power[0]
        bear  = self.eri.bear_power[0]

        if not self.position:
            if bull > 0:
                cash = self.broker.getcash()
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.buy(size=size)
                    msg = f"{dt}: BUY {size} @ {price:.2f} (bull={bull:.2f})"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            if bear < 0:
                sz = self.position.size
                self.sell(size=sz)
                msg = f"{dt}: SELL {sz} @ {price:.2f} (bear={bear:.2f})"
                self.trade_log.append(msg)
                logging.info(msg)

##############################################
# Backtest Runner Function
##############################################
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns,     _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='drawdown')

    logging.info(f"Running {strategy_class.__name__} Strategy...")
    res   = cerebro.run()
    strat = res[0]

    # extract analyzers
    sharpe  = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    returns = strat.analyzers.returns.get_analysis()
    dd_info = strat.analyzers.drawdown.get_analysis()

    # get a numeric max drawdown percentage
    raw_max_dd = dd_info.get(
        'maxdrawdown',
        dd_info.get('max', {}).get('drawdown', dd_info.get('drawdown', 0))
    )
    max_dd = raw_max_dd / 100.0  # convert from % to fraction

    perf = {
        "Sharpe Ratio":      sharpe,
        "Total Return":      returns.get('rtot', 0),
        "Avg Daily Return":  returns.get('ravg', 0),
        "Avg Annual Return": ((1 + returns.get('ravg', 0))**252 - 1),
        "Max Drawdown":      max_dd,
    }

    # plot
    figs = cerebro.plot(iplot=False, show=False)
    fig  = figs[0][0]
    return perf, strat.trade_log, fig

##############################################
# Streamlit App Layout
##############################################
def main():
    st.title("Elder‑Ray Strategy Backtest")

    # sidebar
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start  = st.sidebar.date_input("Start Date",  datetime(2020,1,1).date())
    end    = st.sidebar.date_input("End Date",    datetime.today().date())
    cash   = st.sidebar.number_input("Initial Cash",  value=10000)
    comm   = st.sidebar.number_input("Commission",    value=0.001, step=0.0001)

    # Elder‑Ray params
    ma_p    = st.sidebar.number_input("MA Period",      value=13, step=1)
    ma_t    = st.sidebar.selectbox("MA Type", ["EMA", "SMA"])
    price_c = st.sidebar.selectbox("Price Column", ["close", "open", "high", "low"], index=0)

    if st.sidebar.button("Run Backtest"):
        st.write("Fetching data…")
        data = DataFetcher().get_stock_data(symbol=ticker, start_date=start, end_date=end)
        data_feed = bt.feeds.PandasData(dataname=data, fromdate=start, todate=end)

        st.write("Running backtest…")
        perf, trades, fig = run_backtest(
            strategy_class=ElderRayStrategy,
            data_feed=data_feed,
            cash=cash,
            commission=comm
        )

        st.subheader("Performance Summary")
        st.write(f"**Sharpe Ratio:**      {perf['Sharpe Ratio']:.2f}")
        st.write(f"**Total Return:**      {abs(perf['Total Return'])*100:.2f}%")
        st.write(f"**Avg Daily Return:**  {abs(perf['Avg Daily Return'])*100:.2f}%")
        st.write(f"**Avg Annual Return:** {abs(perf['Avg Annual Return'])*100:.2f}%")
        st.write(f"**Max Drawdown:**      {perf['Max Drawdown']*100:.2f}%")

        st.subheader("Trade Log")
        if trades:
            for t in trades:
                st.write(t)
        else:
            st.write("No trades executed.")

        st.subheader("Backtest Chart")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
