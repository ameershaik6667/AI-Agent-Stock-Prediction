#!/usr/bin/env python3
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib
# Update the system path to import modules from parent directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Use a non-interactive backend for headless environments
matplotlib.use('Agg')
from dotenv import load_dotenv

# Import CrewAI and your agent/indicator modules
from crewai import Crew
# Adjust these import paths as needed in your project:
from src.Indicators.adx_indicator import ADXIndicator  # Your ADX indicator class
from src.UI.adx_main import ADXAnalysisAgent


# Import DataFetcher for fetching price data
from src.Data_Retrieval.data_fetcher import DataFetcher

# Load environment variables
load_dotenv()


#####################################################################
# ADX CrewAI Strategy - using CrewAI decision (BUY/SELL)
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

        # Use the data_df passed in via params
        data_df = self.params.data_df.copy()

        # Calculate ADX indicator on the whole DataFrame.
        adx_indicator = ADXIndicator(period=self.p.adx_period, smoothing_method=self.p.adx_smoothing)
        # Expect the indicator DataFrame to contain columns "ADX", "+DI", and "-DI"
        self.adx_data = adx_indicator.calculate(data_df)

        # Set up CrewAI agent and task for analyzing ADX data.
        adx_agent = ADXAnalysisAgent().adx_investment_advisor()
        adx_analysis_agent = ADXAnalysisAgent()
        # Use the last available close price from the data for the analysis task.
        last_price = data_df['Close'].iloc[-1]
        self.adx_task = adx_analysis_agent.adx_analysis(adx_agent, self.adx_data, current_price=last_price)

        # Run CrewAI agent (this returns a single word: "BUY", "SELL", or "HOLD")
        crew = Crew(
            agents=[adx_agent],
            tasks=[self.adx_task],
            verbose=True
        )
        self.crew_output = crew.kickoff()
        if self.p.printlog:
            print(f"CrewAI output: {self.crew_output}")

    def next(self):
        # Get the latest ADX indicator values (from the previously calculated DataFrame)
        # Here we use the final row of self.adx_data. In practice you might update this over time.
        latest_idx = self.adx_data.index[-1]
        adx_value = self.adx_data.loc[latest_idx, 'ADX']
        plusdi = self.adx_data.loc[latest_idx, '+DI']
        minusdi = self.adx_data.loc[latest_idx, '-DI']
        close_price = self.dataclose[0]

        self.log(f"Close: {close_price:.2f}, ADX: {adx_value:.2f}, +DI: {plusdi:.2f}, -DI: {minusdi:.2f}")

        # Use the precomputed CrewAI decision to trigger trades.
        # (For example, if the CrewAI output is "BUY" and no position exists, generate a buy signal)
        if self.crew_output.upper() == "BUY" and not self.position:
            self.order = self.buy()
            self.log(f"BUY CREATE at {close_price:.2f}")
        elif self.crew_output.upper() == "SELL" and self.position:
            self.order = self.sell()
            self.log(f"SELL CREATE at {close_price:.2f}")

    def log(self, txt, dt=None):
        """ Logging function for this strategy """
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
# Non-CrewAI ADX Strategy - using standard ADX logic (simple rule)
#####################################################################
class ADXStrategy(bt.Strategy):
    params = dict(
        data_df=None,    # Pass the full price DataFrame here
        printlog=True,
        adx_period=14,
        adx_smoothing='SMA',
        adx_threshold=25,
        allocation=1.0
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Calculate ADX on the full DataFrame provided via parameters.
        data_df = self.params.data_df.copy()
        adx_indicator = ADXIndicator(period=self.p.adx_period, smoothing_method=self.p.adx_smoothing)
        self.adx_data = adx_indicator.calculate(data_df)

    def next(self):
        # Get the latest ADX indicator values from pre-computed DataFrame.
        latest_idx = self.adx_data.index[-1]
        adx_value = self.adx_data.loc[latest_idx, 'ADX']
        plusdi = self.adx_data.loc[latest_idx, '+DI']
        minusdi = self.adx_data.loc[latest_idx, '-DI']
        close_price = self.dataclose[0]

        self.log(f"Close: {close_price:.2f}, ADX: {adx_value:.2f}, +DI: {plusdi:.2f}, -DI: {minusdi:.2f}")

        # Buy signal: if +DI > -DI and ADX > threshold and we are not in a position.
        if plusdi > minusdi and adx_value > self.p.adx_threshold and not self.position:
            self.order = self.buy()
            self.log(f"BUY CREATE at {close_price:.2f}")
        # Sell signal: if +DI < -DI and ADX > threshold and we are in a position.
        elif plusdi < minusdi and adx_value > self.p.adx_threshold and self.position:
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
    cerebro.broker.setcommission(commission=0.001)

    # Create the data feed.
    # Make sure the data_df has the required columns. Adjust renames if necessary.
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

    print(f'\nRunning {strategy_name}...')
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_value:.2f}')

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    timereturn = strat.analyzers.timereturn.get_analysis()

    strategy_returns = pd.Series(timereturn)
    cumulative_return = (strategy_returns + 1.0).prod() - 1.0
    start_date = data_df.index[0]
    end_date = data_df.index[-1]
    num_years = (end_date - start_date).days / 365.25
    annual_return = (1 + cumulative_return) ** (1 / num_years) - 1 if num_years != 0 else 0.0

    print(f'\n{strategy_name} Performance Metrics:')
    print('----------------------------------------')
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
    data_fetcher = DataFetcher(start_date=datetime(2015, 1, 1), end_date=datetime(2024, 10, 30))
    data_df = data_fetcher.get_stock_data(company)

    if data_df.empty:
        print(f"No price data found for {company}")
        sys.exit()

    # If there are MultiIndex columns, flatten them.
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [' '.join(col).strip() for col in data_df.columns.values]
        data_df.columns = [col.split(' ')[0] for col in data_df.columns]

    # Rename and filter columns for Backtrader.
    data_df.rename(columns={
        'Adj Close': 'Close',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    data_df = data_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Run ADX CrewAI Strategy.
    adx_metrics_crewai = run_strategy(ADXCrewAIStrategy, 'ADX CrewAI Strategy', data_df, company)

    # Run Non-CrewAI ADX Strategy.
    adx_metrics_noncrewai = run_strategy(ADXStrategy, 'Non-CrewAI ADX Strategy', data_df)

    print("\nComparison of Strategies:")
    print("-------------------------")
    metrics = ['strategy_name', 'sharpe_ratio', 'total_return', 'annual_return', 'max_drawdown']
    df_metrics = pd.DataFrame([adx_metrics_crewai, adx_metrics_noncrewai], columns=metrics)
    print(df_metrics.to_string(index=False))
