#!/usr/bin/env python3
import os
import sys
import pandas as pd
import streamlit as st
import datetime
import yfinance as yf
from dotenv import load_dotenv
from textwrap import dedent

# CrewAI imports for creating and managing agents and tasks
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Import Ticker from yahooquery to fetch the current stock price
from yahooquery import Ticker

# Adjust the system path so that our modules can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ADXIndicator class from the Indicators package
from Indicators.adx_indicator import ADXIndicator

# Load environment variables (e.g., API keys for CrewAI and others)
load_dotenv()

# ---------------------------
# CrewAI Agent Code for ADX Investment Decision
# ---------------------------
# Initialize the GPT model for CrewAI with specified parameters
gpt_model = ChatOpenAI(
    temperature=0.7,  # Lower temperature for more deterministic output
    model_name="gpt-4o"  # Specify the model name
)

class ADXAnalysisAgent:
    def adx_investment_advisor(self):
        """
        Create and return a CrewAI agent that provides investment advice based on ADX indicator data.
        """
        return Agent(
            llm=gpt_model,
            role="ADX Investment Advisor",
            goal="Provide actionable investment recommendations based on ADX indicator data.",
            backstory="You are an experienced technical analyst specializing in the ADX indicator. Analyze the latest ADX values and market conditions to provide a clear decision.",
            verbose=True,
            tools=[]
        )

    def adx_analysis(self, agent, adx_data, current_price):
        """
        Create a task for the CrewAI agent to analyze ADX data and current stock price.
        
        The agent is instructed to return a single word: BUY, SELL, or HOLD.
        
        Parameters:
            agent (Agent): The CrewAI agent instance.
            adx_data (pd.DataFrame): The DataFrame containing ADX calculations.
            current_price (float): The current stock price.
        
        Returns:
            Task: A CrewAI task with a detailed prompt.
        """
        # Extract the latest ADX value from the DataFrame
        latest_adx = adx_data['ADX'].iloc[-1]
        # Build the prompt description with the latest ADX and current stock price
        description = dedent(f"""
            Analyze the latest ADX indicator data:
            - Latest ADX Value: {latest_adx:.2f}
            - Current Stock Price: {current_price}
            
            Based on these values and current market conditions, provide a final investment decision.
            Your final answer must be exactly one of the following words: BUY, SELL, or HOLD.
            Do not include any additional commentary.
        """)
        # Return the task for the agent
        return Task(
            description=description,
            agent=agent,
            expected_output="A single-word decision: BUY, SELL, or HOLD."
        )

# ---------------------------
# Data Fetching Functions
# ---------------------------
def fetch_stock_data(ticker_symbol, period='1y'):
    """
    Fetch historical stock data for a given ticker symbol using yfinance.
    
    Parameters:
        ticker_symbol (str): The stock ticker symbol.
        period (str): The time period for fetching data (e.g., '1y', '6mo').
    
    Returns:
        pd.DataFrame: DataFrame containing historical stock data with columns: Date, High, Low, and Close.
    """
    st.info(f"Fetching historical data for {ticker_symbol} (period={period})...")
    # Download historical data using yfinance
    df = yf.download(ticker_symbol, period=period)
    if df.empty:
        st.error("No data fetched. Please check the ticker symbol or period.")
        return None
    # Reset the index so that the date becomes a column
    df.reset_index(inplace=True)
    # Rename columns to ensure compatibility with our ADX calculation module
    df.rename(columns={'Date': 'Date', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace=True)
    return df

def fetch_realtime_data(ticker_symbol):
    """
    Fetch real-time intraday data for a given ticker symbol using yfinance.
    
    Parameters:
        ticker_symbol (str): The stock ticker symbol.
    
    Returns:
        pd.DataFrame: DataFrame containing real-time intraday data.
    """
    st.info(f"Fetching real-time data for {ticker_symbol}...")
    # Create a Ticker object from yfinance
    ticker = yf.Ticker(ticker_symbol)
    # Download 1-minute interval data for today
    df = ticker.history(period="1d", interval="1m")
    if df.empty:
        st.error("No real-time data fetched. Please check the ticker symbol.")
        return None
    # Reset index to get the date column
    df.reset_index(inplace=True)
    # Rename columns to match the expected format
    df.rename(columns={'Datetime': 'Date', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace=True)
    return df

def fetch_current_price(symbol: str):
    """
    Fetch the current stock price using yahooquery.
    
    Parameters:
        symbol (str): The stock ticker symbol.
    
    Returns:
        float: The current stock price, or None if an error occurs.
    """
    try:
        # Create a Ticker object from yahooquery
        ticker = Ticker(symbol)
        # Retrieve price data from the ticker
        price_data = ticker.price
        # Check if the price data exists and contains the key 'regularMarketPrice'
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ---------------------------
# Main Streamlit Application
# ---------------------------
def main():
    # Set the title and description for the Streamlit app
    st.title("ADX Indicator Calculation and Investment Decision Support")
    st.write("Calculate the ADX indicator for your selected stock with customizable parameters and obtain an investment decision using CrewAI.")
    
    # --- User Input Section ---
    # Input field for the stock ticker symbol
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    # Dropdown for selecting the data period (e.g., 1 year, 6 months)
    data_period = st.selectbox("Select Data Period:", options=["1y", "6mo", "3mo", "1mo"], index=0)
    
    # --- Data Fetching Options ---
    st.subheader("Data Fetching Options")
    # Radio button to choose between historical data and real-time data
    data_type = st.radio("Select Data Type:", options=["Historical Data", "Real-Time Data"])
    
    # Fetch data based on user selection
    if data_type == "Historical Data":
        if st.button("Fetch Historical Data"):
            data = fetch_stock_data(ticker_symbol, period=data_period)
            if data is not None:
                st.subheader(f"Historical Data for {ticker_symbol}")
                st.dataframe(data.tail(10))
                # Store the fetched data in session state for later use
                st.session_state.stock_data = data
    else:
        if st.button("Fetch Real-Time Data"):
            data = fetch_realtime_data(ticker_symbol)
            if data is not None:
                st.subheader(f"Real-Time Data for {ticker_symbol}")
                st.dataframe(data.tail(10))
                # Store the fetched data in session state for later use
                st.session_state.stock_data = data

    # --- ADX Parameter Customization ---
    st.subheader("ADX Indicator Parameters")
    # Allow the user to set the ADX period and smoothing method
    adx_period = st.number_input("ADX Period:", min_value=1, max_value=100, value=14)
    smoothing_method = st.selectbox("Smoothing Method:", options=["SMA", "EMA"])
    
    # Calculate the ADX indicator if stock data has been fetched
    if "stock_data" in st.session_state:
        if st.button("Calculate ADX"):
            # Create an instance of ADXIndicator with the specified parameters
            adx_indicator = ADXIndicator(period=adx_period, smoothing_method=smoothing_method)
            # Compute the ADX values
            adx_data = adx_indicator.calculate(st.session_state.stock_data)
            st.subheader("Stock Data with ADX")
            st.dataframe(adx_data.tail(10))
            # Save the computed ADX data in session state for the decision-making process
            st.session_state.adx_data = adx_data

    # --- Investment Decision via CrewAI ---
    # If ADX data exists, fetch the current price and ask the agent for a decision
    if "adx_data" in st.session_state and 'ADX' in st.session_state.adx_data.columns:
        if st.button("Get Investment Decision"):
            # Retrieve the current stock price using yahooquery
            current_price = fetch_current_price(ticker_symbol)
            if current_price is not None:
                # Instantiate the ADXAnalysisAgent and get the CrewAI advisor agent
                agent_instance = ADXAnalysisAgent()
                advisor_agent = agent_instance.adx_investment_advisor()
                # Create a task instructing the agent to return BUY, SELL, or HOLD
                analysis_task = agent_instance.adx_analysis(advisor_agent, st.session_state.adx_data, current_price)
                # Create a Crew instance with the agent and task, then run it
                crew = Crew(
                    agents=[advisor_agent],
                    tasks=[analysis_task],
                    verbose=True
                )
                # Kick off the CrewAI process and display the result
                result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(result)

if __name__ == '__main__':
    main()
