import pandas as pd
import streamlit as st
from yahooquery import Ticker
from textwrap import dedent
import os
import sys

# CrewAI imports for handling the investment decision support
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# Add parent directory to the system path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the Elder-Ray Analysis Agent from the designated folder
from src.Agents.ElderRay.elder_ray_agent import ElderRayAnalysisAgent


# ---------------------------
# Data Functions
# ---------------------------

def fetch_stock_data(ticker, data_mode, start_date=None, end_date=None, period=None, interval=None):
    """
    Fetch market data using yahooquery.

    Parameters:
    - ticker (str): The stock symbol to fetch data for.
    - data_mode (str): Either "Historical" or "Real-Time" to determine how data is fetched.
    - start_date (datetime, optional): Start date for historical data.
    - end_date (datetime, optional): End date for historical data.
    - period (str, optional): Period for real-time data (e.g., "1d", "5d").
    - interval (str, optional): Data interval for real-time data (e.g., "1m", "15m").

    Returns:
    - pd.DataFrame: The fetched stock data or an empty DataFrame in case of errors.
    """
    try:
        # Instantiate a Ticker object for the given stock symbol
        t = Ticker(ticker)
        if data_mode == "Real-Time":
            # Define allowed intervals for real-time data
            allowed_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
            if interval not in allowed_intervals:
                st.error(f"Interval '{interval}' is not allowed. Allowed values: {allowed_intervals}")
                return pd.DataFrame()
            # Fetch real-time historical data using the specified period and interval
            data = t.history(period=period, interval=interval)
        else:
            # For historical mode, format start and end dates as strings
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            # Fetch historical data with a fixed interval of "1d"
            data = t.history(start=start_str, end=end_str, interval="1d")
        st.write(f"Fetched {len(data)} rows of data.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def calculate_moving_average(series, period, ma_type):
    """
    Calculate a moving average for a given series using either an Exponential Moving Average (EMA)
    or a Simple Moving Average (SMA).

    Parameters:
    - series (pd.Series): The data series on which to calculate the moving average.
    - period (int): The number of periods over which to calculate the average.
    - ma_type (str): The type of moving average to calculate ("EMA" or "SMA").

    Returns:
    - pd.Series: The resulting moving average.
    """
    if ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type == "SMA":
        return series.rolling(window=period).mean()
    else:
        # Default to EMA if an unrecognized type is provided
        return series.ewm(span=period, adjust=False).mean()


def calculate_elder_ray_index(stock_data, ma_period=13, ma_type="EMA", price_column="Close"):
    """
    Calculate the Elder-Ray Index which includes:
      - Bull Power: high price minus the moving average
      - Bear Power: low price minus the moving average

    Parameters:
    - stock_data (pd.DataFrame): DataFrame containing stock price data.
    - ma_period (int): Period for the moving average calculation.
    - ma_type (str): Type of moving average ("EMA" or "SMA").
    - price_column (str): The column name to use for the moving average calculation.

    Returns:
    - pd.DataFrame: The stock_data DataFrame updated with moving average, bull power, and bear power.
    """
    # Standardize column names to lowercase for consistency
    stock_data.columns = [col.lower() for col in stock_data.columns]
    price_column = price_column.lower()
    
    # Calculate the moving average using the specified price column
    stock_data['ma'] = calculate_moving_average(stock_data[price_column], ma_period, ma_type)
    
    # Extract high, moving average, and low series. Handle potential DataFrame wrapping.
    high_series = stock_data['high']
    if isinstance(high_series, pd.DataFrame):
        high_series = high_series.iloc[:, 0]
    ma_series = stock_data['ma']
    if isinstance(ma_series, pd.DataFrame):
        ma_series = ma_series.iloc[:, 0]
    low_series = stock_data['low']
    if isinstance(low_series, pd.DataFrame):
        low_series = low_series.iloc[:, 0]
    
    # Calculate Bull Power: difference between high price and moving average
    stock_data['bull power'] = high_series - ma_series
    # Calculate Bear Power: difference between low price and moving average
    stock_data['bear power'] = low_series - ma_series
    
    return stock_data


def flatten_columns(df):
    """
    Flatten MultiIndex columns in the DataFrame and remove any extraneous columns that start with "index--".
    Reset the DataFrame index after flattening.

    Parameters:
    - df (pd.DataFrame): DataFrame possibly containing MultiIndex columns.

    Returns:
    - pd.DataFrame: DataFrame with flattened columns and reset index.
    """
    # Check if columns are a MultiIndex and flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Filter out columns with names starting with "index--"
    df = df[[col for col in df.columns if not str(col).startswith("index--")]]
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_current_price(symbol):
    """
    Fetch the current stock price using yahooquery.

    Parameters:
    - symbol (str): The stock symbol for which to fetch the current price.

    Returns:
    - float or None: The current market price if successful; otherwise, None.
    """
    try:
        # Instantiate a Ticker object for the given symbol
        t = Ticker(symbol)
        price_data = t.price
        # Check if the price data exists and contains the regularMarketPrice field
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    # Set the title for the Streamlit app
    st.title("Elder-Ray Index Calculator with Investment Decision Support")
    
    # ---------------------------
    # User Inputs
    # ---------------------------
    
    # Input for stock ticker symbol
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", "AAPL")
    # Radio button to select data mode: Historical or Real-Time
    data_mode = st.radio("Select Data Mode", ["Historical", "Real-Time"])
    
    # Moving Average settings: period, type, and which price column to use
    ma_period = st.number_input("Enter Moving Average Period", min_value=1, value=13)
    ma_type = st.selectbox("Select Moving Average Type", ["EMA", "SMA"])
    price_column = st.selectbox("Select Price Column for Moving Average", ["Close", "Open", "High", "Low"], index=0)
    
    # ---------------------------
    # Data Mode Specific Inputs and Processing
    # ---------------------------
    
    if data_mode == "Historical":
        # Date inputs for historical data mode
        start_date = st.date_input("Start Date", pd.to_datetime("2025-02-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2025-03-11"))
        # Validate that the start date is before the end date
        if start_date >= end_date:
            st.error("End Date must be after Start Date.")
            return
        # Button to trigger historical data fetching and Elder-Ray calculation
        if st.button("Calculate Elder-Ray Index (Historical)"):
            with st.spinner("Fetching historical data and calculating Elder-Ray Index..."):
                # Fetch historical stock data
                stock_data = fetch_stock_data(ticker, data_mode, start_date=start_date, end_date=end_date)
                if stock_data.empty:
                    st.error("No data found for the given stock symbol and date range.")
                    return
                # Calculate the Elder-Ray Index from the historical data
                elder_ray_index = calculate_elder_ray_index(stock_data, ma_period, ma_type, price_column)
                # Flatten the DataFrame columns if necessary
                elder_ray_index = flatten_columns(elder_ray_index)
                
                st.subheader(f"Elder-Ray Index for {ticker} from {start_date} to {end_date}")
                # Display the most recent rows of Bull Power and Bear Power values
                st.write(elder_ray_index[["bull power", "bear power"]].tail())
                # Plot the Bull Power and Bear Power as a line chart
                st.line_chart(elder_ray_index[["bull power", "bear power"]])
                
                # Save the computed Elder-Ray data in session state for use in investment decision support
                st.session_state.elder_ray_data = elder_ray_index
    else:
        # For Real-Time mode, input for period and interval
        period = st.text_input("Enter Period for Real-Time Data (e.g., 1d, 5d)", "1d")
        interval = st.text_input("Enter Interval (e.g., 1m, 5m, 15m)", "1m")
        # Button to trigger real-time data fetching and Elder-Ray calculation
        if st.button("Calculate Elder-Ray Index (Real-Time)"):
            with st.spinner("Fetching real-time data and calculating Elder-Ray Index..."):
                # Fetch real-time stock data
                stock_data = fetch_stock_data(ticker, data_mode, period=period, interval=interval)
                if stock_data.empty:
                    st.error("No data found for the given real-time settings.")
                    return
                # Calculate the Elder-Ray Index from the real-time data
                elder_ray_index = calculate_elder_ray_index(stock_data, ma_period, ma_type, price_column)
                # Flatten the DataFrame columns for ease of display
                elder_ray_index = flatten_columns(elder_ray_index)
                
                st.subheader(f"Real-Time Elder-Ray Index for {ticker} (Period: {period}, Interval: {interval})")
                # Display the most recent Bull Power and Bear Power values
                st.write(elder_ray_index[["bull power", "bear power"]].tail())
                # Plot the indicators
                st.line_chart(elder_ray_index[["bull power", "bear power"]])
                
                # Store the computed data in session state for later use
                st.session_state.elder_ray_data = elder_ray_index
    
    # ---------------------------
    # CrewAI Investment Decision Support
    # ---------------------------
    
    # Button to get the investment decision based on the Elder-Ray analysis and current stock price
    if st.button("Get Investment Decision"):
        # Check if Elder-Ray data is available in session state
        if "elder_ray_data" not in st.session_state or st.session_state.elder_ray_data.empty:
            st.error("Please calculate the Elder-Ray Index first.")
        else:
            # Fetch the current stock price for the provided ticker symbol
            current_price = fetch_current_price(ticker)
            if current_price is None:
                st.error("Unable to fetch current stock price.")
            else:
                # Instantiate the Elder-Ray Analysis Agent from the imported module
                agent_obj = ElderRayAnalysisAgent()
                # Retrieve the latest row of the Elder-Ray index data as a string
                last_row = st.session_state.elder_ray_data.tail(1).to_string(index=False)
                # Create a report combining the Elder-Ray data and current price for the investment analysis
                report = dedent(f"""
                    Elder-Ray Analysis Report:
                    {last_row}
                    Current Stock Price: {current_price}
                    
                    Based on the above Elder-Ray index values and the current stock price, please provide 
                    a clear investment recommendation: BUY, SELL, or HOLD. Include supporting reasoning.
                """)
                # Create a task for the CrewAI system to generate an investment recommendation
                decision_task = Task(
                    description=report,
                    agent=agent_obj.elder_ray_investment_advisor(),
                    expected_output="A clear investment recommendation (BUY/SELL/HOLD) with supporting reasoning."
                )
                # Initialize CrewAI with the specified agent and task, enabling verbose output for debugging
                crew = Crew(agents=[agent_obj.elder_ray_investment_advisor()], tasks=[decision_task], verbose=True)
                # Execute the CrewAI task to obtain the decision result
                decision_result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(decision_result)

# Entry point for the Streamlit app
if __name__ == "__main__":
    main()
