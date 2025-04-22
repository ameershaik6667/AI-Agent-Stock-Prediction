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
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.UI.risk_assessment import (
    fetch_stock_data,
    calculate_risk_metrics,
    calculate_scenario_risk_metrics,
    analyze_portfolio_breakdown
)

# ------------------------------
# Backtrader Strategy: Buy & Hold with Risk Metrics
# ------------------------------
class RiskBacktestStrategy(bt.Strategy):
    params = (
        ('confidence', 0.05),
    )
    def __init__(self):
        self.trade_log = []
        self.dates = []
        self.closes = []
        self.bought = False

    def next(self):
        date = self.datas[0].datetime.date(0)
        price = self.datas[0].close[0]
        self.dates.append(date)
        self.closes.append(price)

        if not self.bought:
            size = int(self.broker.getcash() // price)
            if size > 0:
                self.buy(size=size)
                self.trade_log.append(f"{date}: BUY {size} @ {price:.2f}")
            self.bought = True

    def stop(self):
        # Sell at end if still in position
        date = self.datas[0].datetime.date(-1)
        if self.position.size:
            price = self.datas[0].close[-1]
            size = self.position.size
            self.sell(size=size)
            self.trade_log.append(f"{date}: SELL {size} @ {price:.2f}")
        # Compute risk metrics
        df = pd.DataFrame({'date': self.dates, 'close': self.closes})
        self.risk_metrics, self.risk_data = calculate_risk_metrics(df, self.p.confidence)

# ------------------------------
# Backtest Runner Function
# ------------------------------
def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001, confidence=0.05):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, confidence=confidence)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    logging.info(f"Running {strategy_class.__name__}...")
    results = cerebro.run()
    strat = results[0]

    # Performance metrics
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', np.nan)
    total_return = strat.analyzers.returns.get_analysis().get('rtot', 0)
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_dd = drawdown.get('maxdrawdown', np.nan)
    trades = strat.analyzers.trades.get_analysis().get('total', {})

    perf_summary = {
        'Sharpe Ratio': sharpe,
        'Total Return': total_return,
        'Max Drawdown': max_dd
    }

    # Risk metrics
    risk_metrics = strat.risk_metrics
    # Trade log
    trade_log = strat.trade_log

    # Chart
    figs = cerebro.plot(iplot=False, show=False)
    chart = figs[0][0]

    return perf_summary, risk_metrics, trade_log, chart

# ------------------------------
# Streamlit App Layout
# ------------------------------
def main():
    st.title("Risk Assessment Backtest")
    st.write(
        "Backtest a buy-and-hold strategy, compute performance (Sharpe, return, drawdown), "
        "risk metrics (VaR, volatility), visualize results, simulate shock, and analyze portfolio."
    )

    # Sidebar
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    period = st.sidebar.selectbox("Period", ["1y", "6mo", "3mo", "1mo"], index=0)
    confidence = st.sidebar.slider("VaR Confidence", 0.01, 0.1, 0.05, 0.01)
    shock_pct = st.sidebar.slider("Market Shock (%)", 0.0, 10.0, 0.0, 0.5)
    portfolio_input = st.sidebar.text_area(
        "Portfolio (Ticker,Asset Class,Size)", "AAPL,Equity,100\nMSFT,Equity,150"
    )
    cash = st.sidebar.number_input("Initial Cash", 10000.0)
    commission = st.sidebar.number_input("Commission", 0.001, step=0.0001)

    if st.sidebar.button("Run Backtest"):
        st.info("Fetching data...")
        data = fetch_stock_data(ticker, period)
        if data is None:
            st.error("Data fetch failed.")
            return
        data_feed = bt.feeds.PandasData(dataname=data, datetime='date', close='close')

        st.info("Running backtest...")
        perf, risk, trades, chart = run_backtest(
            RiskBacktestStrategy,
            data_feed,
            cash=cash,
            commission=commission,
            confidence=confidence
        )

        # Performance
        st.subheader("Performance Summary")
        st.write(f"**Sharpe Ratio:** {perf['Sharpe Ratio']:.2f}")
        st.write(f"**Total Return:** {perf['Total Return']*100:.2f}%")
        st.write(f"**Max Drawdown:** {perf['Max Drawdown']:.2%}")

        # Risk
        st.subheader("Risk Metrics")
        st.write(f"**VaR ({confidence*100:.0f}%):** {risk['var']:.2%}")
        st.write(f"**Volatility:** {risk['volatility']:.2%}")

        # Trades
        st.subheader("Trade Log")
        if trades:
            for t in trades:
                st.write(t)
        else:
            st.write("No trades.")

        # Chart
        st.subheader("Backtest Chart")
        st.pyplot(chart)

        # Scenario
        if shock_pct > 0:
            scene = calculate_scenario_risk_metrics(pd.DataFrame({'date': risk['returns'].index, 'close': risk['returns'].index}), shock_pct/100.0, confidence)
            st.subheader("Scenario Analysis")
            st.write(f"**VaR (shock {shock_pct}%):** {scene['var']:.2%}")
            st.write(f"**Volatility:** {scene['volatility']:.2%}")

        # Portfolio
        st.subheader("Portfolio Breakdown")
        details, breakdown = analyze_portfolio_breakdown(portfolio_input, period, confidence)
        if details is not None:
            st.dataframe(details)
            st.dataframe(breakdown)
        else:
            st.warning("No portfolio data.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
