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
import numpy as np
import pandas as pd
import os
import sys

# Update the system path to import modules from parent directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import your data fetching module.
from src.Data_Retrieval.data_fetcher import DataFetcher

##############################################
# Ichimoku Calculator (for indicator calculation)
##############################################
# This calculator closely follows your original Ichimoku calculation logic.
class IchimokuCalculator:
    def __init__(self, df, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26, smoothing_factor=1):
        self.df = df.copy()  # Work on a copy to avoid modifying the input DataFrame.
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        self.smoothing_factor = smoothing_factor

    def calculate(self):
        # If a date column is available, sort the DataFrame.
        if 'date' in self.df.columns:
            self.df.sort_values(by='date', inplace=True)
        
        # Tenkan-sen (Conversion Line)
        self.df['tenkan_sen'] = (
            self.df['high'].rolling(window=self.tenkan_period, min_periods=self.tenkan_period).max() +
            self.df['low'].rolling(window=self.tenkan_period, min_periods=self.tenkan_period).min()
        ) / 2

        # Kijun-sen (Base Line)
        self.df['kijun_sen'] = (
            self.df['high'].rolling(window=self.kijun_period, min_periods=self.kijun_period).max() +
            self.df['low'].rolling(window=self.kijun_period, min_periods=self.kijun_period).min()
        ) / 2

        # Senkou Span A (Leading Span A)
        self.df['senkou_span_a'] = ((self.df['tenkan_sen'] + self.df['kijun_sen']) / 2).shift(self.displacement)

        # Senkou Span B (Leading Span B)
        self.df['senkou_span_b'] = (
            self.df['high'].rolling(window=self.senkou_b_period, min_periods=self.senkou_b_period).max() +
            self.df['low'].rolling(window=self.senkou_b_period, min_periods=self.senkou_b_period).min()
        ) / 2
        self.df['senkou_span_b'] = self.df['senkou_span_b'].shift(self.displacement)

        # Chikou Span (Lagging Span)
        self.df['chikou_span'] = self.df['close'].shift(-self.displacement)

        # Optional smoothing if needed.
        if self.smoothing_factor > 1:
            self.df['tenkan_sen'] = self.df['tenkan_sen'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['kijun_sen'] = self.df['kijun_sen'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['senkou_span_a'] = self.df['senkou_span_a'].rolling(window=self.smoothing_factor, min_periods=1).mean()
            self.df['senkou_span_b'] = self.df['senkou_span_b'].rolling(window=self.smoothing_factor, min_periods=1).mean()

        return self.df

##############################################
# Ichimoku Indicator Wrapper for Backtrader
##############################################
class IchimokuIndicatorBT(bt.Indicator):
    """
    A Backtrader Indicator that calculates the Ichimoku Cloud components:
      - Tenkan-sen (Conversion Line)
      - Kijun-sen (Base Line)
      - Senkou Span A (Leading Span A)
      - Senkou Span B (Leading Span B)
      - Chikou Span (Lagging Span)
    """
    lines = ('tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',)
    params = (
        ('tenkan_period', 9),
        ('kijun_period', 26),
        ('senkou_b_period', 52),
        ('displacement', 26),
        ('smoothing_factor', 1),
    )

    def __init__(self):
        # Set minimum period: we need enough bars to calculate the longest lookback plus displacement.
        min_period = max(self.p.kijun_period, self.p.senkou_b_period) + self.p.displacement
        self.addminperiod(min_period)

    def once(self, start, end):
        size = self.data.buflen()
        # Build a DataFrame from the Backtrader data feed.
        df = pd.DataFrame({
            'high': [self.data.high[i] for i in range(size)],
            'low': [self.data.low[i] for i in range(size)],
            'close': [self.data.close[i] for i in range(size)],
        })
        # Create a dummy date column so the calculator can sort if needed.
        df['date'] = pd.date_range(end=datetime.today(), periods=size, freq='D')
        
        # Calculate Ichimoku Cloud indicators.
        calc = IchimokuCalculator(df,
                                  tenkan_period=self.p.tenkan_period,
                                  kijun_period=self.p.kijun_period,
                                  senkou_b_period=self.p.senkou_b_period,
                                  displacement=self.p.displacement,
                                  smoothing_factor=self.p.smoothing_factor)
        result = calc.calculate()

        # Assign computed indicator values to the corresponding Backtrader lines.
        for i in range(size):
            if i < len(result):
                self.lines.tenkan_sen[i] = result['tenkan_sen'].iloc[i]
                self.lines.kijun_sen[i] = result['kijun_sen'].iloc[i]
                self.lines.senkou_span_a[i] = result['senkou_span_a'].iloc[i]
                self.lines.senkou_span_b[i] = result['senkou_span_b'].iloc[i]
                self.lines.chikou_span[i] = result['chikou_span'].iloc[i]
            else:
                self.lines.tenkan_sen[i] = 0
                self.lines.kijun_sen[i] = 0
                self.lines.senkou_span_a[i] = 0
                self.lines.senkou_span_b[i] = 0
                self.lines.chikou_span[i] = 0

##############################################
# Ichimoku Strategy
##############################################
class IchimokuStrategy(bt.Strategy):
    """
    A simple strategy based on Ichimoku signals.
    
    Entry Rule:
      - When not in a position and the current closing price is above the cloud (i.e., above both Senkou Span A and Senkou Span B),
        a long position is initiated.
        
    Exit Rule:
      - When in a position and the current closing price falls below the cloud (i.e., below either Senkou Span A or Senkou Span B),
        the position is closed.
      
    All trade events are recorded in self.trade_log.
    """
    params = (
        ('tenkan_period', 9),
        ('kijun_period', 26),
        ('senkou_b_period', 52),
        ('displacement', 26),
        ('smoothing_factor', 1),
        ('allocation', 1.0),
    )

    def __init__(self):
        self.trade_log = []
        self.ichi = IchimokuIndicatorBT(self.data,
                                        tenkan_period=self.p.tenkan_period,
                                        kijun_period=self.p.kijun_period,
                                        senkou_b_period=self.p.senkou_b_period,
                                        displacement=self.p.displacement,
                                        smoothing_factor=self.p.smoothing_factor)

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        close = self.data.close[0]

        # Determine the cloud boundaries.
        senkou_a = self.ichi.senkou_span_a[0]
        senkou_b = self.ichi.senkou_span_b[0]
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if not self.position:
            # Entry: Buy if the closing price is above the cloud.
            if close > cloud_top:
                cash = self.broker.getcash()
                size = int((cash * self.p.allocation) // close)
                if size > 0:
                    self.buy(size=size)
                    msg = f"{current_date}: BUY {size} shares at {close:.2f}"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            # Exit: Sell if the closing price falls below the cloud.
            if close < cloud_bottom:
                size = self.position.size
                self.sell(size=size)
                msg = f"{current_date}: SELL {size} shares at {close:.2f}"
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

    # Add analyzers for performance metrics.
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    logging.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()
    strat = result[0]

    # Extract metrics.
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown_duration = drawdown.get('maxdrawdownperiod', 'N/A')

    perf_summary = {
        "Sharpe Ratio": sharpe.get('sharperatio', 0),
        "Total Return": returns.get('rtot', 0),
        "Avg Daily Return": returns.get('ravg', 0),
        "Avg Annual Return": ((1+returns.get('ravg', 0))**252 - 1),
        "Max Drawdown": drawdown.drawdown,
        "Max Drawdown Duration": max_drawdown_duration
    }

    # Plot the backtest results.
    figs = cerebro.plot(iplot=False, show=False)
    fig = figs[0][0]
    return perf_summary, strat.trade_log, fig

##############################################
# Streamlit App Layout
##############################################
def main():
    st.title("Ichimoku Strategy Backtest")

    # Sidebar: Input parameters.
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1).date())
    end_date = st.sidebar.date_input("End Date", value=datetime.today().date())
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000)
    commission = st.sidebar.number_input("Commission", value=0.001, step=0.0001)

    # Ichimoku indicator parameters.
    tenkan_period = st.sidebar.number_input("Tenkan-sen Period", value=9, step=1)
    kijun_period = st.sidebar.number_input("Kijun-sen Period", value=26, step=1)
    senkou_b_period = st.sidebar.number_input("Senkou Span B Period", value=52, step=1)
    displacement = st.sidebar.number_input("Displacement", value=26, step=1)
    smoothing_factor = st.sidebar.number_input("Smoothing Factor", value=1, step=1)

    if st.sidebar.button("Run Backtest"):
        st.write("Fetching data...")
        data = DataFetcher().get_stock_data(symbol=ticker, start_date=start_date, end_date=end_date)
        data_feed = bt.feeds.PandasData(dataname=data, fromdate=start_date, todate=end_date)

        st.write("Running backtest. Please wait...")
        perf_summary, trade_log, fig = run_backtest(
            strategy_class=IchimokuStrategy,
            data_feed=data_feed,
            cash=initial_cash,
            commission=commission
        )

        st.subheader("Performance Summary")
        st.write(f"**Sharpe Ratio:** {perf_summary['Sharpe Ratio']:.2f}")
        st.write(f"**Total Return:** {perf_summary['Total Return']*100:.2f}%")
        st.write(f"**Avg Daily Return:** {perf_summary['Avg Daily Return']*100:.2f}%")
        st.write(f"**Avg Annual Return:** {perf_summary['Avg Annual Return']*100:.2f}%")
        st.write(f"**Max Drawdown:** {perf_summary['Max Drawdown']*100:.2f}%")
        st.write(f"**Max Drawdown Duration:** {perf_summary['Max Drawdown Duration']}")

        st.subheader("Trade Log")
        if trade_log:
            for t in trade_log:
                st.write(t)
        else:
            st.write("No trades executed.")

        st.subheader("Backtest Chart")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
