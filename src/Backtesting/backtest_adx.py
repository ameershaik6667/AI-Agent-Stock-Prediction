import os
# Force Matplotlib to use the non-interactive Agg backend (required for Streamlit)
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
# Ensure Matplotlib uses Agg even if another backend was configured
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
# Switch the pyplot interface to Agg backend
plt.switch_backend("agg")

import streamlit as st            # Streamlit for web UI
from datetime import datetime    # Working with date inputs
import logging                   # Logging trades and events
import backtrader as bt          # Backtesting framework
import numpy as np               # Numeric operations
import pandas as pd              # Data manipulation
import os                        # os functions (imported twice but kept as original)
import sys                       # System path modification

# Update the system path so we can import modules from project parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Custom data fetcher and ADX indicator calculation
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.adx_indicator import ADXIndicator

# Configure global logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

#######################################
# ADX Indicator Wrapper for Backtrader
#######################################
class ADXIndicatorBT(bt.Indicator):
    """
    Backtrader Indicator wrapper for custom ADXIndicator.
    Computes +DI, -DI, and ADX once enough bars are available.
    """
    # Define output lines for +DI, -DI, and ADX
    lines = ('di_plus', 'di_minus', 'adx',)
    # Parameter specifying lookback period for ADX
    params = (('period', 14),)

    def __init__(self):
        # Require at least 'period' bars before producing values
        self.addminperiod(self.p.period)

    def once(self, start, end):
        """
        After all data is loaded, convert Backtrader feed to DataFrame,
        calculate ADX series, and populate the indicator lines.
        """
        size = self.data.buflen()  # total number of bars loaded

        # Build a pandas DataFrame for High, Low, Close
        df = pd.DataFrame({
            'High':  [self.data.high[i]  for i in range(size)],
            'Low':   [self.data.low[i]   for i in range(size)],
            'Close': [self.data.close[i] for i in range(size)]
        })

        # Run the custom ADX calculation
        adx_calc = ADXIndicator(period=self.p.period)
        result = adx_calc.calculate(df)

        # Map results back to Backtrader lines for each bar
        for i in range(size):
            self.lines.di_plus[i]  = result['+DI'].iloc[i] if i < len(result) else 0
            self.lines.di_minus[i] = result['-DI'].iloc[i] if i < len(result) else 0
            self.lines.adx[i]      = result['ADX'].iloc[i] if i < len(result) else 0

#######################################
# ADX Strategy
#######################################
class ADXStrategy(bt.Strategy):
    """
    Simple ADX-based trend-following strategy:
    - Enter long when ADX > threshold (strong trend)
    - Exit when ADX falls below threshold (trend weakens)
    """
    params = (
        ('period', 14),           # ADX lookback period
        ('adx_threshold', 25),    # threshold for strong trend
        ('allocation', 1.0),      # fraction of cash to allocate
    )

    def __init__(self):
        # List to record human-readable trade log messages
        self.trade_log = []

        # Attach custom ADX indicator
        self.adx_indicator = ADXIndicatorBT(self.data, period=self.p.period)
        self.adx = self.adx_indicator.adx

    def next(self):
        """
        Called on every new bar:
        - If not in position and ADX > threshold, buy
        - If in position and ADX < threshold, sell
        """
        current_date = self.datas[0].datetime.date(0)

        # ENTRY logic
        if not self.position:
            if self.adx[0] > self.p.adx_threshold:
                cash = self.broker.getcash()
                price = self.data.close[0]
                # Determine number of shares to buy
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.buy(size=size)
                    msg = f"{current_date}: BUY {size} shares at {price:.2f}"
                    self.trade_log.append(msg)
                    logging.info(msg)

        # EXIT logic
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
    """
    Configure and execute the backtest, then collect performance metrics,
    trade log, and the resulting matplotlib figure.
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    cerebro.adddata(data_feed)

    # Set broker starting cash and commission rate
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    # Add performance analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    logging.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()      # execute backtest
    strat = result[0]           # get the strategy instance

    # Extract analyzer results
    sharpe   = strat.analyzers.sharpe.get_analysis()
    returns  = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown_duration = drawdown.get('maxdrawdownperiod', 'N/A')

    # Build performance summary
    perf_summary = {
        "Sharpe Ratio":          sharpe.get('sharperatio', 0),
        "Total Return":          returns.get('rtot', 0),
        "Avg Daily Return":      returns.get('ravg', 0),
        "Avg Annual Return":     ((1 + returns.get('ravg', 0)) ** 252 - 1),
        "Max Drawdown":          drawdown.drawdown,
        "Max Drawdown Duration": max_drawdown_duration
    }

    # Generate plot without displaying it immediately
    figs = cerebro.plot(iplot=False, show=False)
    fig = figs[0][0]  # Extract the first figure

    return perf_summary, strat.trade_log, fig

#######################################
# Streamlit App Layout
#######################################
def main():
    """
    Streamlit UI:
    - Sidebar for user inputs
    - Run backtest on button click
    - Display performance, trade log, and plot
    """
    st.title("ADX Strategy Backtest")

    # Sidebar inputs for backtest parameters
    st.sidebar.header("Backtest Parameters")
    ticker        = st.sidebar.text_input("Ticker", value="SPY")
    start_date    = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1).date())
    end_date      = st.sidebar.date_input("End Date", value=datetime.today().date())
    initial_cash  = st.sidebar.number_input("Initial Cash", value=10000)
    commission    = st.sidebar.number_input("Commission", value=0.001, step=0.0001)
    adx_threshold = st.sidebar.number_input("ADX Threshold", value=25)
    adx_period    = st.sidebar.number_input("ADX Period", value=14, step=1)

    # Run backtest when user clicks the button
    if st.sidebar.button("Run Backtest"):
        st.write("Fetching data...")
        data = DataFetcher().get_stock_data(symbol=ticker,
                                            start_date=start_date,
                                            end_date=end_date)
        data_feed = bt.feeds.PandasData(dataname=data,
                                        fromdate=start_date,
                                        todate=end_date)

        st.write("Running backtest. Please wait...")
        perf_summary, trade_log, fig = run_backtest(
            strategy_class=ADXStrategy,
            data_feed=data_feed,
            cash=initial_cash,
            commission=commission
        )

        # Display performance metrics
        st.subheader("Performance Summary")
        st.write(f"**Sharpe Ratio:** {perf_summary['Sharpe Ratio']:.2f}")
        st.write(f"**Total Return:** {perf_summary['Total Return']*100:.2f}%")
        st.write(f"**Avg Daily Return:** {perf_summary['Avg Daily Return']*100:.2f}%")
        st.write(f"**Avg Annual Return:** {perf_summary['Avg Annual Return']*100:.2f}%")
        st.write(f"**Max Drawdown:** {perf_summary['Max Drawdown']*100:.2f}%")
        st.write(f"**Max Drawdown Duration:** {perf_summary['Max Drawdown Duration']}")

        # Show logged trades
        st.subheader("Trade Log")
        if trade_log:
            for t in trade_log:
                st.write(t)
        else:
            st.write("No trades executed.")

        # Plot the backtest chart
        st.subheader("Backtest Chart")
        st.pyplot(fig)

if __name__ == '__main__':
    # Entry point for script execution
    main()
