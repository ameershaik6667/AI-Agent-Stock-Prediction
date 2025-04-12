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
# Import your modules – adjust these imports per your project structure.
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.adx_indicator import ADXIndicator


# Set up logging.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#######################################
# ADX Indicator Wrapper for Backtrader
#######################################
class ADXIndicatorBT(bt.Indicator):
    """
    A Backtrader Indicator that wraps your ADX calculation.
    It computes +DI, -DI, and ADX over the full dataset.
    
    Note: It requires a minimum of 'period' bars before outputting valid values.
    """
    lines = ('di_plus', 'di_minus', 'adx',)
    params = (('period', 14),)  # ADX period

    def __init__(self):
        # Require that at least 'period' bars have been loaded.
        self.addminperiod(self.p.period)

    def once(self, start, end):
        size = self.data.buflen()
        # Build a DataFrame from the Backtrader data feed.
        df = pd.DataFrame({
            'High': [self.data.high[i] for i in range(size)],
            'Low':  [self.data.low[i] for i in range(size)],
            'Close': [self.data.close[i] for i in range(size)]
        })

        # Calculate ADX-related indicators using your custom ADXIndicator.
        adx_calc = ADXIndicator(period=self.p.period)
        result = adx_calc.calculate(df)

        # Set indicator lines from the resulting DataFrame.
        for i in range(size):
            self.lines.di_plus[i] = result['+DI'].iloc[i] if i < len(result) else 0
            self.lines.di_minus[i] = result['-DI'].iloc[i] if i < len(result) else 0
            self.lines.adx[i] = result['ADX'].iloc[i] if i < len(result) else 0

#######################################
# ADX Strategy
#######################################
class ADXStrategy(bt.Strategy):
    """
    A simple ADX-based strategy.
    
    Entry Rule:
      - When not in a position and ADX exceeds a defined threshold (indicating a strong trend),
        enter a long position.
    
    Exit Rule:
      - When in a position and ADX falls below the threshold (indicating the trend is weakening),
        exit the position.
    
    All trade events are stored in self.trade_log for later display.
    """
    params = (
        ('period', 14),
        ('adx_threshold', 25),  # Threshold for a strong trend.
        ('allocation', 1.0),
    )

    def __init__(self):
        # Initialize a log list to store trades.
        self.trade_log = []
        # Attach the ADX indicator.
        self.adx_indicator = ADXIndicatorBT(self.data, period=self.p.period)
        self.adx = self.adx_indicator.adx

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        if not self.position:
            if self.adx[0] > self.p.adx_threshold:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.buy(size=size)
                    msg = f"{current_date}: BUY {size} shares at {price:.2f}"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            if self.adx[0] < self.p.adx_threshold:
                size = self.position.size
                price = self.data.close[0]
                self.sell(size=size)
                msg = f"{current_date}: SELL {size} shares at {price:.2f}"
                self.trade_log.append(msg)
                logging.info(msg)

#######################################
# Backtest Runner Function
#######################################
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    # Remove runonce=True and preload=True to let next() run on each bar.
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add analyzers.
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    logging.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()
    strat = result[0]

    # Extract analyzer outputs.
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

    # Use cerebro.plot with iplot=False and show=False.
    figs = cerebro.plot(iplot=False, show=False)
    # Backtrader returns a list of lists; select the first figure.
    fig = figs[0][0]
    return perf_summary, strat.trade_log, fig

#######################################
# Streamlit App Layout
#######################################
def main():
    st.title("ADX Strategy Backtest")

    # Sidebar for parameters.
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1).date())
    end_date = st.sidebar.date_input("End Date", value=datetime.today().date())
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000)
    commission = st.sidebar.number_input("Commission", value=0.001, step=0.0001)
    adx_threshold = st.sidebar.number_input("ADX Threshold", value=25)
    adx_period = st.sidebar.number_input("ADX Period", value=14, step=1)

    if st.sidebar.button("Run Backtest"):
        st.write("Fetching data...")
        data = DataFetcher().get_stock_data(symbol=ticker, start_date=start_date, end_date=end_date)
        data_feed = bt.feeds.PandasData(dataname=data, fromdate=start_date, todate=end_date)

        st.write("Running backtest. Please wait...")
        perf_summary, trade_log, fig = run_backtest(
            strategy_class=ADXStrategy,
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