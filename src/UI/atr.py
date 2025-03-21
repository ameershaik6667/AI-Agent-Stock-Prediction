import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
import os
import sys

# Adjust the system path so that our modules can be imported (if needed).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# For fetching current price using yahooquery
from yahooquery import Ticker

# ----- CrewAI Integration Imports -----
from crewai import Agent, Task, Crew
from textwrap import dedent
from langchain_openai import ChatOpenAI

# Import tools for CrewAI (ensure these modules exist in your project)
from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools
from langchain_community.tools import YahooFinanceNewsTool

# Optionally load environment variables (e.g., API keys)
# from dotenv import load_dotenv
# load_dotenv()

# ----- Data Fetching and ATR Calculation Functions -----
def fetch_historical_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical stock data for the specified date range and interval.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

def fetch_realtime_data(ticker):
    """
    Fetch near real-time data (past 1 day, 1-minute intervals).
    """
    realtime_data = yf.download(ticker, period='1d', interval='1m')
    return realtime_data

def calculate_atr(stock_data, period=14):
    """
    Calculate the Average True Range (ATR) for the given stock data and period.
    """
    data = stock_data.copy()
    # Calculate the True Range components
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift(1))
    # True Range is the maximum of the three values
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    # ATR is the rolling average of the True Range
    data['ATR'] = data['True Range'].rolling(window=period).mean()
    return data

def fetch_current_price(symbol: str):
    """
    Fetch the current stock price for the given symbol using yahooquery's ticker.price.
    """
    try:
        ticker = Ticker(symbol)
        price_data = ticker.price
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ----- CrewAI ATR Analysis Agent -----
# Initialize the LLM (e.g., OpenAI GPT model)
gpt_model = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4"
)

class ATRAnalysisAgents:
    def atr_investment_advisor(self):
        """
        Returns an agent that analyzes ATR data to provide actionable investment advice.
        """
        return Agent(
            llm=gpt_model,
            role="ATR Investment Advisor",
            goal="Provide actionable investment recommendations based on ATR indicator data and current stock price.",
            backstory=(
                "You are an experienced technical analyst specializing in using the ATR indicator to gauge market "
                "volatility and guide investment decisions. Analyze the latest ATR values along with the current "
                "stock price to determine optimal trade entries, exits, and position sizing."
            ),
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
    
    def atr_analysis(self, agent, atr_data, current_price):
        """
        Creates a task for the agent to analyze the latest ATR data and current stock price,
        providing a clear investment decision.
        """
        # Extract the latest ATR value
        latest_atr = atr_data['ATR'].dropna().iloc[-1] if not atr_data['ATR'].dropna().empty else None
        description = dedent(f"""
            Analyze the latest ATR indicator data along with the current stock price.
            Latest ATR Value: {latest_atr}
            Current Stock Price: {current_price}
            
            Based on these values and the current market conditions, provide a detailed investment recommendation.
            Your final answer should clearly indicate whether to BUY, SELL, or HOLD, along with supporting reasoning.
        """)
        return Task(
            description=description,
            agent=agent,
            expected_output="A detailed analysis and a clear buy, sell, or hold recommendation based on the ATR data and current stock price."
        )

# ----- Streamlit UI -----
st.title("Stock Data and ATR Calculator with Real-Time and CrewAI Integration")

# 1. User input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")

# 2. User-selectable date range for Historical Data
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.date(2022, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.date.today())

# 3. Interval selection for Historical Data
interval_options = ["1d", "1wk", "1mo"]
selected_interval = st.selectbox("Select Data Interval (Historical)", options=interval_options, index=0)

# 4. ATR Period slider
atr_period = st.slider("ATR Period", min_value=1, max_value=60, value=14, step=1)

# 5. ATR Alert Threshold
atr_threshold = st.number_input(
    "ATR Alert Threshold (optional)",
    min_value=0.0,
    value=0.0,
    step=1.0,
    help="If the ATR exceeds this value, a warning will be displayed."
)

# 6. Columns to Display
available_columns = ["Open", "High", "Low", "Close", "Volume", "ATR"]
selected_columns = st.multiselect(
    "Select Columns to Display",
    options=available_columns,
    default=["High", "Low", "Close", "ATR"]
)

# ----- Data Fetching Section -----
st.subheader("Data Fetching")

# Button to Fetch Historical Data
if st.button('Fetch Historical Data'):
    if ticker:
        historical_data = fetch_historical_data(ticker, start_date, end_date, selected_interval)
        if not historical_data.empty:
            st.session_state.historical_data = historical_data
            st.write(f"Historical data for {ticker} (first 5 rows):")
            st.write(historical_data.head())
        else:
            st.warning("No historical data returned. Please check your date range or ticker symbol.")
    else:
        st.error("Please enter a valid stock ticker.")

# Button to Fetch Real-Time Data
if st.button('Fetch Real-Time Data'):
    if ticker:
        realtime_data = fetch_realtime_data(ticker)
        if not realtime_data.empty:
            st.session_state.realtime_data = realtime_data
            st.write(f"Real-Time data for {ticker} (last 5 rows):")
            st.write(realtime_data.tail())
        else:
            st.warning("No real-time data returned. The provider may not have recent data for this ticker.")
    else:
        st.error("Please enter a valid stock ticker.")

# ----- ATR Calculation Section -----
st.subheader("ATR Calculation")

# Option to select which dataset to use for ATR
dataset_choice = st.radio(
    "Choose the dataset for ATR calculation:",
    ("Historical Data", "Real-Time Data")
)

if st.button('Calculate ATR'):
    # Determine which dataset to use
    data_key = 'historical_data' if dataset_choice == "Historical Data" else 'realtime_data'
    
    # Check if the chosen dataset is available
    if data_key in st.session_state and not st.session_state[data_key].empty:
        # Calculate ATR using the chosen dataset and user-defined period
        df_atr = calculate_atr(st.session_state[data_key], period=atr_period)

        # Check if ATR has exceeded the threshold and display a warning if so
        latest_atr = df_atr['ATR'].dropna().iloc[-1] if not df_atr['ATR'].dropna().empty else 0.0
        if atr_threshold > 0 and latest_atr > atr_threshold:
            st.warning(f"Alert: The current ATR ({latest_atr:.2f}) exceeds the threshold of {atr_threshold}")

        # Display the last 5 rows of selected columns
        st.write(f"Displaying {dataset_choice} and ATR (period={atr_period}) for {ticker}:")
        columns_to_show = [col for col in selected_columns if col in df_atr.columns]
        if columns_to_show:
            st.write(df_atr[columns_to_show].tail())
        else:
            st.info("No columns selected to display.")

        # Plot ATR if requested
        if "ATR" in columns_to_show:
            st.line_chart(df_atr["ATR"].dropna())
        
        # Save the calculated ATR data to session state for further processing
        st.session_state.atr_data = df_atr
    else:
        st.error(f"No {dataset_choice.lower()} available. Please fetch the data first.")

# ----- CrewAI Investment Decision Support Section -----
st.subheader("CrewAI Investment Decision Support")

# Button to fetch the current stock price
if st.button("Fetch Current Stock Price"):
    if ticker:
        current_price = fetch_current_price(ticker)
        if current_price is not None:
            st.session_state.current_price = current_price
            st.write(f"Current Stock Price for {ticker}: {current_price}")
    else:
        st.error("Please enter a valid stock ticker.")

# Button to get the investment decision from CrewAI agents based on ATR data and current price
if st.button("Get Investment Decision"):
    if "atr_data" not in st.session_state:
        st.error("Please calculate the ATR first.")
    elif "current_price" not in st.session_state:
        st.error("Please fetch the current stock price first.")
    else:
        atr_data = st.session_state.atr_data
        current_price = st.session_state.current_price
        # Instantiate the ATR analysis agent
        agents = ATRAnalysisAgents()
        advisor_agent = agents.atr_investment_advisor()
        # Create an analysis task using ATR data and the current stock price
        analysis_task = agents.atr_analysis(advisor_agent, atr_data, current_price)
        # Create a CrewAI instance and kickoff the analysis task
        crew = Crew(
            agents=[advisor_agent],
            tasks=[analysis_task],
            verbose=True
        )
        result = crew.kickoff()
        st.subheader("Investment Decision")
        st.write(result)
