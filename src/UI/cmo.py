#!/usr/bin/env python3
import sys
import os
import pandas as pd
import streamlit as st
import yfinance as yf
from yahooquery import Ticker
from dotenv import load_dotenv
from textwrap import dedent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# CrewAI imports (ensure these modules are available in your project)
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools
from langchain_community.tools import YahooFinanceNewsTool

# Load environment variables and adjust system path
load_dotenv()


# ---------------------------------------------
# Data Fetching Functions
def fetch_data_yfinance(ticker_symbol, data_source='Historical', period='1y', interval='1d'):
    """
    Fetch stock data using yfinance for either Historical or Real-Time (intraday) data.
    
    Parameters:
        ticker_symbol: Stock symbol (e.g., 'AAPL').
        data_source: 'Historical' or 'Real-Time'.
        period: Lookback period (e.g., '1y', '6mo') for historical data.
        interval: Data interval (e.g., '1d', '1m').
        
    Returns:
        DataFrame with columns: date, high, low, close.
    """
    if data_source == 'Historical':
        st.info(f"Fetching historical data for {ticker_symbol} (period={period}, interval={interval})...")
        data = yf.download(ticker_symbol, period=period, interval=interval)
    else:
        st.info(f"Fetching real-time data for {ticker_symbol} (interval={interval})...")
        data = yf.download(ticker_symbol, period='1d', interval=interval)
    
    if data is None or data.empty:
        st.error("No data returned from yfinance.")
        return None

    data.reset_index(inplace=True)
    rename_map = {
        'Date': 'date',
        'Datetime': 'date',
        'Close': 'close',
        'High': 'high',
        'Low': 'low'
    }
    data.rename(columns=rename_map, inplace=True)
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            st.error(f"Column '{col}' is missing in the data.")
            return None
    return data

def fetch_current_price(symbol: str):
    """
    Fetch the current stock price for a given symbol using yahooquery.
    
    Parameters:
        symbol: Stock symbol.
    
    Returns:
        The current stock price or None.
    """
    try:
        ticker = Ticker(symbol)
        price_data = ticker.price
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Current price not found in ticker.price data.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ---------------------------------------------
# Highlighting Function
def highlight_cmo_above_threshold(val, threshold):
    """Highlight cell text red if CMO value is above the threshold, otherwise black."""
    color = 'red' if val >= threshold else 'black'
    return f'color: {color}'

# ---------------------------------------------
# CMO Calculation Class
class CMOCalculator:
    """
    Computes the Chande Momentum Oscillator (CMO) from closing prices.
    
    Calculation:
      - Price differences are computed from consecutive closing prices.
      - Gains: Positive differences (or absolute values if 'Absolute' is chosen).
      - Losses: Negative differences, made positive.
      - CMO = 100 * (sum(gains) - sum(losses)) / (sum(gains) + sum(losses))
    """
    def __init__(self, df, period=14, calc_method='Standard', apply_smoothing=None, smoothing_period=3, keep_intermediate=False):
        """
        Parameters:
            df: DataFrame with at least a 'close' column.
            period: Lookback period.
            calc_method: 'Standard' (default) or 'Absolute'.
            apply_smoothing: Optional, either 'SMA' or 'EMA'.
            smoothing_period: Period for smoothing.
            keep_intermediate: If True, keep intermediate calculation columns.
        """
        self.df = df.copy()
        self.period = period
        self.calc_method = calc_method
        self.apply_smoothing = apply_smoothing
        self.smoothing_period = smoothing_period
        self.keep_intermediate = keep_intermediate

    def calculate(self):
        self.df['price_change'] = self.df['close'].diff()
        if self.calc_method == 'Absolute':
            self.df['gain'] = self.df['price_change'].abs()
            self.df['loss'] = 0
        else:
            self.df['gain'] = self.df['price_change'].where(self.df['price_change'] > 0, 0)
            self.df['loss'] = -self.df['price_change'].where(self.df['price_change'] < 0, 0)
        self.df['gain_sum'] = self.df['gain'].rolling(window=self.period).sum()
        self.df['loss_sum'] = self.df['loss'].rolling(window=self.period).sum()
        self.df['cmo'] = 100 * (self.df['gain_sum'] - self.df['loss_sum']) / (self.df['gain_sum'] + self.df['loss_sum'])
        if self.apply_smoothing == 'SMA':
            self.df['cmo'] = self.df['cmo'].rolling(window=self.smoothing_period).mean()
        elif self.apply_smoothing == 'EMA':
            self.df['cmo'] = self.df['cmo'].ewm(span=self.smoothing_period, adjust=False).mean()
        if not self.keep_intermediate:
            self.df.drop(columns=['price_change', 'gain', 'loss', 'gain_sum', 'loss_sum'], inplace=True)
        return self.df

# ---------------------------------------------
# Crew AI Agent for CMO-based Investment Decision
chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o")

class CMOAnalysisAgent:
    def create_agent(self):
        """Creates a Crew AI agent for CMO-based recommendations."""
        return Agent(
            llm=chat_model,
            role="CMO Investment Advisor",
            goal="Provide a BUY, SELL, or HOLD recommendation based on the latest CMO and current stock price.",
            backstory="You are a seasoned technical analyst specializing in momentum indicators. Analyze the latest CMO value along with the current stock price and market conditions to deliver a clear investment recommendation.",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k,
                YahooFinanceNewsTool()
            ]
        )

    def create_task(self, agent, cmo_df, current_price):
        """Creates a task for the agent with the latest CMO value and current stock price."""
        latest_cmo = cmo_df['cmo'].iloc[-1]
        task_description = dedent(f"""
            Please analyze the following data and provide a definitive investment recommendation:
            - Current Stock Price: {current_price}
            - Latest CMO Value: {latest_cmo}
            
            Based on the above, decide if the stock should be BUY, SELL, or HOLD. Provide your reasoning.
        """)
        return Task(
            description=task_description,
            agent=agent,
            expected_output="A clear BUY, SELL, or HOLD recommendation with supporting analysis."
        )

# ---------------------------------------------
# Main Streamlit Application
def main():
    st.title("Chande Momentum Oscillator (CMO) Calculation System")
    st.write("Fetch Historical/Real-Time Data, calculate the CMO, and get investment decisions via Crew AI.")

    # Data Source Selection
    data_source = st.radio(
        "Select Data Source:",
        options=["Historical", "Real-Time"],
        index=0,
        help="Choose Historical data or Real-Time (intraday) data from yfinance."
    )
    if data_source == "Historical":
        period_option = st.selectbox("Historical Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)
        interval_option = st.selectbox("Historical Interval:", options=["1d", "1wk", "1mo"], index=0)
        st.info("Historical data will be retrieved.")
    else:
        period_option = '1d'
        interval_option = st.selectbox("Intraday Interval:", options=["1m", "5m", "15m", "30m", "1h"], index=0)
        st.warning("Real-time quotes may be delayed; they are treated as intraday data.")

    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

    # CMO Customization Options
    st.subheader("CMO Parameters")
    cmo_period = st.number_input("CMO Calculation Period:", min_value=1, max_value=200, value=14)
    calc_method = st.selectbox("Gains/Loss Calculation Method:", options=["Standard", "Absolute"], index=0)
    apply_smoothing = st.selectbox("Apply Smoothing to CMO?", options=[None, "SMA", "EMA"],
                                   format_func=lambda x: "None" if x is None else x, index=0)
    smoothing_period = st.number_input("Smoothing Period (for SMA/EMA):", min_value=1, max_value=50, value=3)
    keep_intermediate = st.checkbox("Keep intermediate columns for debugging?", value=False)
    threshold_enable = st.checkbox("Enable threshold highlight on CMO?", value=False)
    threshold_value = st.number_input("Highlight CMO above threshold value:", value=70) if threshold_enable else None

    # Button: Fetch & Calculate CMO
    if st.button("Fetch & Calculate CMO"):
        data = fetch_data_yfinance(ticker_symbol, data_source=data_source, period=period_option, interval=interval_option)
        if data is not None:
            st.subheader(f"Fetched Data for {ticker_symbol} ({data_source})")
            st.dataframe(data.tail(10))
            cmo_calc = CMOCalculator(
                df=data,
                period=cmo_period,
                calc_method=calc_method,
                apply_smoothing=apply_smoothing,
                smoothing_period=smoothing_period,
                keep_intermediate=keep_intermediate
            )
            cmo_results = cmo_calc.calculate()
            st.subheader("Calculated CMO Data")
            if threshold_enable and threshold_value is not None:
                styled_cmo = cmo_results.style.applymap(
                    lambda x: highlight_cmo_above_threshold(x, threshold_value),
                    subset=['cmo']
                )
                st.dataframe(styled_cmo)
            else:
                st.dataframe(cmo_results.tail(20))
            st.session_state.cmo_results = cmo_results

    # Button: Get Investment Decision using Crew AI Agent
    if st.button("Get Investment Decision"):
        if "cmo_results" not in st.session_state:
            st.error("Please calculate the CMO first.")
        else:
            current_price = fetch_current_price(ticker_symbol)
            if current_price is None:
                st.error("Unable to fetch current stock price.")
            else:
                cmo_results = st.session_state.cmo_results
                advisor = CMOAnalysisAgent()
                agent = advisor.create_agent()
                task = advisor.create_task(agent, cmo_results, current_price)
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                decision = crew.kickoff()  # Execute the Crew AI workflow
                st.subheader("Investment Decision")
                st.write(decision)

if __name__ == '__main__':
    main()
