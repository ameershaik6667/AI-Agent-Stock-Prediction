#!/usr/bin/env python3
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import yfinance as yf
import matplotlib
# Update the system path to import modules from parent directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Use a non-interactive backend for headless environments:
matplotlib.use('Agg')
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your ADX CrewAI agent (adjust the import path as needed)
from src.UI.adx_main import ADXAnalysisAgent  # ADXAnalysisAgent is defined here

# Import your ADX indicator
from src.Indicators.adx_indicator import ADXIndicator

# Import your DataFetcher (adjust the import path as needed)
from src.Data_Retrieval.data_fetcher import DataFetcher


#####################################################################
# ADX CrewAI Strategy - using CrewAI decision (BUY/SELL/HOLD)
#####################################################################
class ADXCrewAIStrategy(bt.Strategy):
    params = dict(
        company='AAPL',
        data_df=None,   # Pass the full price DataFrame here
        printlog=True,
        adx_period=14,
        adx_smoothing='SMA',
        adx_threshold=25  # Only trade if ADX indicates a strong trend
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Use the full DataFrame passed in via params.
        data_df = self.params.data_df.copy()

        # Calculate ADX indicator on the entire DataFrame.
        adx_indicator = ADXIndicator(period=self.p.adx_period, smoothing_method=self.p.adx_smoothing)
        self.adx_data = adx_indicator.calculate(data_df)

        # Set up CrewAI agent and task for analyzing ADX data.
        adx_agent = ADXAnalysisAgent().adx_investment_advisor()
        adx_analysis_agent = ADXAnalysisAgent()
        # Use the most recent close from the DataFrame (or adjust as needed)
        last_price = data_df['Close'].iloc[-1]
        self.adx_task = adx_analysis_agent.adx_analysis(adx_agent, self.adx_data, current_price=last_price)

        # Run CrewAI agent (expected to return a single word: BUY, SELL, or HOLD)
        from crewai import Crew  # Ensure Crew is imported here
        crew = Crew(
            agents=[adx_agent],
            tasks=[self.adx_task],
            verbose=True
        )
        self.crew_output = crew.kickoff()
        if self.p.printlog:
            print(f"CrewAI output: {self.crew_output}")

    def next(self):
        # Use the full calculated ADX data. (You might update this on each new bar if needed.)
        latest_idx = self.adx_data.index[-1]
        adx_value = self.adx_data.loc[latest_idx, 'ADX']
        plusdi = self.adx_data.loc[latest_idx, '+DI']
        minusdi = self.adx_data.loc[latest_idx, '-DI']
        close_price = self.dataclose[0]

        self.log(f"Close: {close_price:.2f}, ADX: {adx_value:.2f}, +DI: {plusdi:.2f}, -DI: {minusdi:.2f}")

        # Convert the CrewAI output to a string before using .upper()
        decision = str(self.crew_output).upper()
        # Use the decision to trigger trades.
        if decision == "BUY" and not self.position:
            self.order = self.buy()
            self.log(f"BUY CREATE at {close_price:.2f}")
        elif decision == "SELL" and self.position:
            self.order = self.sell()
            self.log(f"SELL CREATE at {close_price:.2f}")
        # If decision is HOLD, no action is taken.

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")


#####################################################################
# Non-CrewAI MACD Strategy - using standard MACD rules
#####################################################################
class MACDStrategy(bt.Strategy):
    params = dict(
        data_df=None,   # Full price DataFrame passed as parameter
        printlog=True,
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Use the full DataFrame passed in via params.
        data_df = self.params.data_df.copy()
        # Import your MACDIndicator from your MACD module (adjust the path as needed)
        from src.Indicators.macd import MACDIndicator
        macd = MACDIndicator(data_df)
        self.macd_data = macd.calculate_macd()

    def next(self):
        macd_value = self.macd_data['MACD'].iloc[-1]
        signal_value = self.macd_data['Signal_Line'].iloc[-1]
        close_price = self.dataclose[0]

        self.log(f"MACD: {macd_value:.2f}, Signal: {signal_value:.2f}, Close: {close_price:.2f}")

        # Buy if MACD crosses above Signal Line and no position exists.
        if macd_value > signal_value and not self.position:
            self.order = self.buy()
            self.log(f"BUY CREATE at {close_price:.2f}")
        # Sell if MACD crosses below Signal Line and a position exists.
        elif macd_value < signal_value and self.position:
            self.order = self.sell()
            self.log(f"SELL CREATE at {close_price:.2f}")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {order.executed.price:.2f}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")


#####################################################################
# Backtesting Runner Function
#####################################################################
def run_strategy(strategy_class, strategy_name, data_df, company=None):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    # For demonstration you might adjust commission as needed:
    cerebro.broker.setcommission(commission=0.2)

    # Create the data feed—assuming data_df contains columns: Open, High, Low, Close, Volume.
    data = bt.feeds.PandasData(
        dataname=data_df,
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)

    # Pass data_df (and company if applicable) to the strategy.
    if 'company' in strategy_class.params._getkeys():
        cerebro.addstrategy(strategy_class, data_df=data_df, company=company, printlog=True)
    else:
        cerebro.addstrategy(strategy_class, data_df=data_df, printlog=True)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, _name='timereturn')

    print(f"\nRunning {strategy_name}...")
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {final_value:.2f}")

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    strategy_returns = pd.Series(timereturn)
    cumulative_return = (strategy_returns + 1.0).prod() - 1.0
    start_date = data_df.index[0]
    end_date = data_df.index[-1]
    num_years = (end_date - start_date).days / 365.25
    annual_return = (1 + cumulative_return) ** (1 / num_years) - 1 if num_years != 0 else 0.0

    print(f"\n{strategy_name} Performance Metrics:")
    print("----------------------------------------")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Total Return: {cumulative_return * 100:.2f}%")
    print(f"Annual Return: {annual_return * 100:.2f}%")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")

    return {
        'strategy_name': strategy_name,
        'sharpe_ratio': sharpe.get('sharperatio', 'N/A'),
        'total_return': cumulative_return * 100,
        'annual_return': annual_return * 100,
        'max_drawdown': drawdown.max.drawdown,
    }


#####################################################################
# Main Execution
#####################################################################
if __name__ == '__main__':
    company = 'AAPL'
    # Fetch historical price data using DataFetcher.
    data_fetcher = DataFetcher(start_date=datetime(2024, 1, 1), end_date=datetime(2024, 12, 31))
    data_df = data_fetcher.get_stock_data(company)

    if data_df.empty:
        print(f"No price data found for {company}")
        sys.exit()

    # Flatten MultiIndex columns if necessary.
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [' '.join(col).strip() for col in data_df.columns.values]
        data_df.columns = [col.split(' ')[0] for col in data_df.columns]

    # Rename columns for Backtrader.
    data_df.rename(columns={
        'Adj Close': 'Close',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Note: yfinance now uses auto_adjust=True by default.
    # Run ADX CrewAI Strategy.
    adx_metrics_crewai = run_strategy(ADXCrewAIStrategy, 'ADX CrewAI Strategy', data_df, company)

    # Run Non-CrewAI MACD Strategy.
    macd_metrics_noncrewai = run_strategy(MACDStrategy, 'Non-CrewAI MACD Strategy', data_df)

    print("\nComparison of Strategies:")
    print("-------------------------")
    metrics = ['strategy_name', 'sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown']
    df_metrics = pd.DataFrame([adx_metrics_crewai, macd_metrics_noncrewai], columns=metrics)
    print(df_metrics.to_string(index=False))
