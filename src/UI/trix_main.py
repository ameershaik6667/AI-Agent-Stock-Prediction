import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import datetime
import yfinance as yf

# Additional imports for CrewAI integration
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from textwrap import dedent
from yahooquery import Ticker  # Used for fetching current price via yahooquery

# Import your existing modules
from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.trix import calculate_trix

# ---------------------------------------------------------
# Function to Fetch the Current Stock Price using yahooquery
# ---------------------------------------------------------
def fetch_current_price(symbol: str):
    """
    Fetch the current stock price for the given symbol using yahooquery's ticker.price.
    
    Parameters:
        symbol (str): Stock symbol to fetch the current price for.
    
    Returns:
        The current stock price if successful, otherwise None.
    """
    try:
        ticker_yq = Ticker(symbol)
        price_data = ticker_yq.price
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ---------------------------------------------------------
# CrewAI Agent Setup for TRIX Investment Decision Support
# ---------------------------------------------------------

# Initialize the ChatOpenAI model for CrewAI
gpt_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o")

class TrixInvestmentAdvisor:
    def advisor_agent(self):
        """
        Returns a CrewAI agent configured to provide investment recommendations based on TRIX data.
        """
        return Agent(
            llm=gpt_model,
            role="Trix Investment Advisor",
            goal="Provide actionable investment recommendations based on TRIX indicator data and current stock price.",
            backstory=(
                "You are an experienced technical analyst specialized in TRIX indicator analysis. "
                "Analyze the latest TRIX values along with the current stock price and market conditions "
                "to provide a clear buy, sell, or hold signal with supporting reasoning."
            ),
            verbose=True,
            tools=[]  # Additional tools can be added if needed
        )
    
    def analyze_trix_and_price(self, agent, trix_data, current_price):
        """
        Creates a task for the agent to analyze TRIX data and the current stock price
        to generate an investment decision.
        """
        latest_trix = trix_data['TRIX'].iloc[-1]
        # If TRIX_SIGNAL exists, get the latest value; otherwise, set it to None.
        latest_trix_signal = trix_data['TRIX_SIGNAL'].iloc[-1] if 'TRIX_SIGNAL' in trix_data.columns else None
        
        # Construct the analysis description
        description = dedent(f"""
            Analyze the following TRIX indicator data along with the current stock price:
            - Latest TRIX: {latest_trix}
            - Latest TRIX Signal: {latest_trix_signal}
            - Current Stock Price: {current_price}
            
            Based on these values and current market conditions, provide a detailed investment recommendation.
            Clearly indicate whether to BUY, SELL, or HOLD, and provide supporting reasoning.
        """)
        return Task(
            description=description,
            agent=agent,
            expected_output="A detailed analysis with a clear buy, sell, or hold recommendation based on TRIX data and the current stock price."
        )

# ---------------------------------------------------------
# Main Streamlit Application: TRIX Indicator & Investment Decision Support
# ---------------------------------------------------------
st.title("TRIX Indicator: Customization, Data Integration & Investment Decision Support")

# 1. User Input for Stock Symbol
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

# 2. Get the Current Stock Price using yfinance for display purposes
try:
    ticker = yf.Ticker(symbol)
    current_price_yfinance = ticker.info.get("regularMarketPrice")
    if current_price_yfinance is not None:
        st.write(f"Current Stock Price for {symbol}: ${current_price_yfinance:.2f}")
    else:
        st.write("Current stock price not available from yfinance.")
except Exception as e:
    st.error(f"Error retrieving current stock price from yfinance: {e}")

# 3. Select Data Source: Real-Time or Historical
data_source_choice = st.radio(
    "Select Data Source:",
    options=["Real-Time", "Historical"],
    index=0
)

# 4. Optional Date Range Selection for Historical Data
start_date = None
end_date = None
if data_source_choice == "Historical":
    st.write("Select the date range for historical data:")
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=30)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=default_start)
    with col2:
        end_date = st.date_input("End Date", value=today)
    if start_date > end_date:
        st.warning("Start date cannot be after end date. Please adjust your selection.")

# 5. Fetch Stock Data using the Existing DataFetcher
data_fetcher = DataFetcher()
if data_source_choice == "Real-Time":
    # Fetch intraday data for today using yfinance and filter for today's records only
    data = ticker.history(period="1d", interval="1m")
    today_date = datetime.date.today()
    data = data[data.index.date == today_date]
else:
    # Historical data fetch with date range using the existing DataFetcher
    data = data_fetcher.get_stock_data(symbol, start_date=start_date, end_date=end_date)

st.write(f"Showing data for: {symbol}")
st.dataframe(data.tail())

# 6. Inputs for TRIX Calculation (Customization Features)
st.subheader("TRIX Indicator Parameters")
trix_length = st.number_input("TRIX Length (EMA periods):", min_value=1, max_value=100, value=14, key="trix_length")
trix_signal = st.number_input("TRIX Signal Period:", min_value=1, max_value=100, value=9, key="trix_signal")
apply_additional_smoothing = st.checkbox("Apply Additional Smoothing?")  # Optional smoothing toggle

# 7. Calculate and Display TRIX Values
if st.button("Calculate TRIX"):
    # Make a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    # Compute TRIX using the user-defined parameters
    data_with_trix = calculate_trix(data_copy, length=trix_length, signal=trix_signal)
    
    # Optionally apply additional smoothing to the TRIX output
    if apply_additional_smoothing:
        data_with_trix['TRIX_SMOOTHED'] = data_with_trix['TRIX'].ewm(span=5, adjust=False).mean()
    
    st.write(f"TRIX Calculation Results for {symbol}:")
    st.dataframe(data_with_trix.tail())
    
    # Validate that TRIX values are computed correctly
    if data_with_trix['TRIX'].isna().all():
        st.warning("All TRIX values are NaN. Check data range or historical data availability.")
    else:
        st.success("TRIX calculation complete. Parameters successfully applied.")
    
    # Store the calculated TRIX data in session state for use by CrewAI agents
    st.session_state.trix_data = data_with_trix

# 8. Get Investment Decision using CrewAI Agents
if st.button("Get Investment Decision"):
    # Ensure TRIX data has been calculated first
    if "trix_data" not in st.session_state:
        st.error("Please calculate TRIX values first.")
    else:
        # Fetch the current stock price using the provided function
        current_price = fetch_current_price(symbol)
        if current_price is None:
            st.error("Unable to fetch current stock price for decision making.")
        else:
            trix_data = st.session_state.trix_data
            # Initialize the TRIX Investment Advisor CrewAI agent
            advisor = TrixInvestmentAdvisor()
            agent = advisor.advisor_agent()
            # Create a task for the agent using the latest TRIX data and current stock price
            task = advisor.analyze_trix_and_price(agent, trix_data, current_price)
            # Create a Crew with the agent and task, and run the CrewAI process
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True
            )
            result = crew.kickoff()
            st.subheader("Investment Decision")
            st.write(result)

