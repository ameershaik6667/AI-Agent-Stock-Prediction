#!/usr/bin/env python3
import os
import sys
import pandas as pd
import streamlit as st
from yahooquery import Ticker
from dotenv import load_dotenv
from datetime import date
from textwrap import dedent

# CrewAI and LLM imports
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import additional tools for CrewAI (ensure these modules exist in your project)
from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools
from langchain_community.tools import YahooFinanceNewsTool

# For live data auto-refresh (if needed)
from streamlit_autorefresh import st_autorefresh

# Load environment variables if needed (e.g., API keys)
load_dotenv()

# -------------------------
# Data Fetching Functions
# -------------------------
def fetch_stock_data(ticker_symbol, start_date=None, end_date=None):
    """
    Fetch historical stock data for a given ticker symbol using yahooquery.
    If start_date and end_date are provided, data is fetched between those dates.
    Ensures that the DataFrame contains the required columns: date, high, low, and close.
    """
    st.info(f"Fetching historical data for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    if start_date and end_date:
        data = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
    else:
        data = ticker.history(period='1y')
    
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
        # Convert 'date' to datetime if not already
        data['date'] = pd.to_datetime(data['date'])
    else:
        st.error("Failed to fetch data as a DataFrame.")
        return None
    
    # Ensure required columns exist; rename if necessary.
    for col in ['date', 'high', 'low', 'close']:
        if col not in data.columns:
            if col.capitalize() in data.columns:
                data.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                st.error(f"Required column '{col}' not found in data.")
                return None
    return data

def fetch_realtime_data(ticker_symbol):
    """
    Fetch current market data for a given ticker symbol using yahooquery.
    Returns a DataFrame with the current market data.
    """
    st.info(f"Fetching real-time data for {ticker_symbol}...")
    ticker = Ticker(ticker_symbol)
    try:
        realtime_data = ticker.price
        if realtime_data:
            df_rt = pd.DataFrame([realtime_data])
            return df_rt
        else:
            st.error("Failed to fetch real-time data.")
            return None
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return None

def fetch_current_price(symbol: str):
    """
    Fetch the current stock price for the given symbol using yahooquery's ticker.price.
    
    Parameters:
        symbol: Stock symbol to fetch the current price for.
    
    Returns:
        The current stock price if successful, otherwise None.
    """
    try:
        ticker = Ticker(symbol)
        price_data = ticker.price
        # Ensure the structure contains the required information
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# -------------------------
# Donchian Channels Calculator
# -------------------------
class DonchianCalculator:
    """
    Calculates the Donchian Channels indicator:
      - donchian_high: The highest high over the specified look-back period.
      - donchian_low: The lowest low over the specified look-back period.
    """
    def __init__(self, df, window=20):
        self.df = df.copy()
        self.window = window

    def calculate(self):
        # Sort the DataFrame by date if available.
        if 'date' in self.df.columns:
            self.df.sort_values(by='date', inplace=True)
        # Calculate the highest high and lowest low over the given window.
        self.df['donchian_high'] = self.df['high'].rolling(window=self.window, min_periods=self.window).max()
        self.df['donchian_low'] = self.df['low'].rolling(window=self.window, min_periods=self.window).min()
        return self.df

# -------------------------
# CrewAI Agent for Investment Decision
# -------------------------
# Initialize the ChatOpenAI model for CrewAI
gpt_model = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o"
)

class DonchianAnalysisAgents:
    """
    Provides CrewAI agents that analyze the Donchian Channels indicator data
    and current stock price to provide actionable investment advice.
    """
    def donchian_investment_advisor(self):
        """
        Returns an agent configured as a Donchian Channels Investment Advisor.
        """
        return Agent(
            llm=gpt_model,
            role="Donchian Channels Investment Advisor",
            goal="Provide actionable investment recommendations based on Donchian Channels indicator data and the current stock price.",
            backstory=dedent("""
                You are an experienced technical analyst specializing in Donchian Channels.
                Analyze the latest indicator values (the highest high and lowest low) along with the current stock price.
                Provide clear BUY, SELL, or HOLD recommendations with supporting reasoning.
            """),
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

    def donchian_analysis(self, agent, donchian_data, current_price):
        """
        Creates a task for the agent to analyze the latest Donchian Channels indicator data
        along with the current stock price and provide an investment recommendation.
        """
        latest_donchian_high = donchian_data['donchian_high'].iloc[-1]
        latest_donchian_low = donchian_data['donchian_low'].iloc[-1]

        description = dedent(f"""
            Analyze the latest Donchian Channels indicator data:
            - Donchian High (highest high over the look-back period): {latest_donchian_high}
            - Donchian Low (lowest low over the look-back period): {latest_donchian_low}
            - Current Stock Price: {current_price}

            Based on these values and current market conditions, provide a detailed investment recommendation.
            Your final answer should clearly indicate whether to BUY, SELL, or HOLD, along with supporting reasoning.
        """)
        return Task(
            description=description,
            agent=agent,
            expected_output="A detailed analysis and a clear buy, sell, or hold recommendation based on the Donchian Channels data and the current stock price."
        )

# -------------------------
# Main Streamlit Application
# -------------------------
def main():
    st.title("Donchian Channels Indicator & Investment Decision System")
    st.write("Fetch stock data, calculate the Donchian Channels indicator, and receive an investment decision from CrewAI agents.")

    # -------------------------
    # Sidebar Inputs and Buttons
    # -------------------------
    st.sidebar.header("Configuration")
    # Stock ticker symbol input
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL")
    
    # Data type selection: Historical or Real-Time
    data_type = st.sidebar.radio("Select Data Type", options=["Historical Data", "Real-Time Data"])
    
    # Historical data options: start and end dates
    if data_type == "Historical Data":
        st.sidebar.subheader("Historical Data Options")
        start_date = st.sidebar.date_input("Start Date", value=date(2022, 1, 1))
        end_date = st.sidebar.date_input("End Date", value=date.today())
    else:
        st.sidebar.info("Real-Time Data: Recent data with 1-minute intervals will be fetched.")
        start_date = None
        end_date = None

    # Button to fetch stock data
    if st.sidebar.button("Fetch Stock Data"):
        if data_type == "Historical Data":
            data = fetch_stock_data(ticker_symbol, start_date, end_date)
        else:
            data = fetch_realtime_data(ticker_symbol)
        if data is not None:
            st.session_state["data"] = data
            st.success("Stock data fetched successfully!")
        else:
            st.error("Failed to fetch stock data.")

    # Display fetched stock data if available
    if "data" in st.session_state and st.session_state["data"] is not None:
        st.subheader("Fetched Stock Data")
        st.dataframe(st.session_state["data"].tail(10))

    # Donchian Channel customization options
    st.sidebar.subheader("Donchian Channel Customization")
    donchian_window = st.sidebar.number_input("Donchian Window (Look-back Period)", min_value=1, max_value=100, value=20)

    # Button to calculate the Donchian Channels indicator
    if st.sidebar.button("Calculate Donchian Channels Indicator"):
        if "data" in st.session_state and st.session_state["data"] is not None:
            calc = DonchianCalculator(st.session_state["data"], window=donchian_window)
            data_with_channels = calc.calculate()
            st.session_state["data"] = data_with_channels
            st.success("Donchian Channels calculated successfully!")
        else:
            st.error("No stock data available. Please fetch the stock data first.")

    # Display calculated Donchian Channels data if available
    if ("data" in st.session_state and st.session_state["data"] is not None and 
        "donchian_high" in st.session_state["data"].columns):
        st.subheader("Stock Data with Donchian Channels")
        st.dataframe(st.session_state["data"].tail(20))

    # -------------------------
    # CrewAI Investment Decision Section
    # -------------------------
    # Button to get investment decision from CrewAI
    if st.sidebar.button("Get Investment Decision"):
        # Ensure the Donchian Channels indicator has been calculated
        if "data" not in st.session_state or st.session_state["data"] is None or "donchian_high" not in st.session_state["data"].columns:
            st.error("Please fetch and calculate the Donchian Channels indicator first.")
        else:
            # Fetch the current stock price
            current_price = fetch_current_price(ticker_symbol)
            if current_price is None:
                st.error("Failed to fetch current stock price. Cannot proceed with investment decision.")
            else:
                donchian_data = st.session_state["data"]
                # Initialize the CrewAI agent for Donchian Channels investment advice
                agents = DonchianAnalysisAgents()
                advisor_agent = agents.donchian_investment_advisor()
                # Create a task for the agent with the latest indicator values and current price
                analysis_task = agents.donchian_analysis(advisor_agent, donchian_data, current_price)
                # Create a Crew with the agent and task, and kick off the analysis
                crew = Crew(
                    agents=[advisor_agent],
                    tasks=[analysis_task],
                    verbose=True
                )
                result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(result)

if __name__ == '__main__':
    main()
