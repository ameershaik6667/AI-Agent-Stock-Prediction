#!/usr/bin/env python3
import os
import sys
import pandas as pd
import streamlit as st
from yahooquery import Ticker  # For fetching stock data using yahooquery
import yfinance as yf       # For fetching stock data using yfinance
from dotenv import load_dotenv  # To load environment variables from a .env file
import matplotlib.pyplot as plt  # For plotting charts

# Add the parent directories to the system path so that we can import modules from src/Agents/VPT
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the VPTAnalysisAgent from the separate file in src/Agents/VPT
from src.Agents.VPT.vpt_agent import VPTAnalysisAgent

# Load environment variables if available
load_dotenv()

# ---------------------------
# Data Fetching and VPT Calculation Functions
# ---------------------------
def fetch_historical_data(ticker_symbol, period='1y', start_date=None, end_date=None):
    """
    Fetch historical stock data for a given ticker symbol.
    Uses yfinance for custom date ranges or yahooquery for preset periods.
    
    Parameters:
        ticker_symbol (str): The stock symbol to fetch data for.
        period (str): The period of historical data to fetch (e.g., '1y').
        start_date (date, optional): The start date for custom date range.
        end_date (date, optional): The end date for custom date range.
    
    Returns:
        pd.DataFrame: DataFrame containing the historical stock data with required columns.
    """
    st.info(f"Fetching historical data for {ticker_symbol}...")
    
    # Check if custom start and end dates are provided
    if start_date and end_date:
        # Use yfinance with the custom date range
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
    else:
        # Use yahooquery for preset periods
        ticker = Ticker(ticker_symbol)
        data = ticker.history(period=period)
    
    # Ensure the fetched data is a DataFrame
    if isinstance(data, pd.DataFrame):
        data.reset_index(inplace=True)
    else:
        st.error("Failed to fetch historical data as a DataFrame.")
        return None

    # Convert column names to lowercase for consistency
    data.columns = data.columns.str.lower()
    
    # Ensure that the required columns exist in the data
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in historical data: {', '.join(missing_columns)}")
        return None
    return data

def fetch_realtime_data(ticker_symbol, period='1d', interval='1m'):
    """
    Fetch real-time stock data for a given ticker symbol using yfinance.
    
    Parameters:
        ticker_symbol (str): The stock symbol to fetch data for.
        period (str): The period to fetch real-time data (e.g., '1d').
        interval (str): The data interval (e.g., '1m').
    
    Returns:
        pd.DataFrame: DataFrame containing the real-time stock data with required columns.
    """
    st.info(f"Fetching real-time data for {ticker_symbol} (period={period}, interval={interval})...")
    
    # Download real-time data using yfinance
    data = yf.download(ticker_symbol, period=period, interval=interval)
    if data.empty:
        st.error("Failed to fetch real-time data.")
        return None
    data.reset_index(inplace=True)
    
    # Handle multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Convert column names to lowercase for consistency
    data.columns = data.columns.str.lower()
    
    # Rename 'datetime' column to 'date' if necessary, or set the index name appropriately
    if 'datetime' in data.columns:
        data.rename(columns={'datetime': 'date'}, inplace=True)
    elif 'date' not in data.columns and data.index.name is not None:
        data.index.name = 'date'
        data.reset_index(inplace=True)
    
    # Ensure that the required columns exist in the data
    required_columns = ['date', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns in real-time data: {', '.join(missing_columns)}")
        return None
    return data

def calculate_vpt(stock_data, calc_period=1, weighting_factor=1.0, apply_smoothing=False, smoothing_window=5):
    """
    Calculate the Volume Price Trend (VPT) indicator for the stock data.
    
    Parameters:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        calc_period (int): Period over which to calculate percentage change in price.
        weighting_factor (float): A multiplier to adjust the impact of volume.
        apply_smoothing (bool): Whether to apply a rolling average smoothing to the VPT values.
        smoothing_window (int): Window size for the rolling average.
    
    Returns:
        pd.DataFrame: Stock data with additional columns for price change and VPT.
    """
    # Calculate percentage change in close price
    stock_data['Price Change %'] = stock_data['close'].pct_change(periods=calc_period)
    
    # Calculate VPT as the cumulative sum of volume * percentage change * weighting factor
    stock_data['VPT'] = (stock_data['volume'] * stock_data['Price Change %'] * weighting_factor).cumsum()
    
    # Optionally apply smoothing to the VPT using a rolling average
    if apply_smoothing and smoothing_window > 1:
        stock_data['VPT'] = stock_data['VPT'].rolling(window=int(smoothing_window), min_periods=1).mean()
    return stock_data

def fetch_current_price(symbol: str):
    """
    Fetch the current market price for a given stock symbol using yahooquery.
    
    Parameters:
        symbol (str): The stock symbol.
    
    Returns:
        float: The current market price or None if fetching fails.
    """
    try:
        ticker = Ticker(symbol)
        price_data = ticker.price
        # Check if the price data is available and contains the 'regularMarketPrice'
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    # Set the title and description of the Streamlit app
    st.title("Volume Price Trend (VPT) Indicator with Investment Decision Support")
    st.write("Customize, calculate the VPT indicator, and get an investment decision based on VPT and current stock price.")

    # Input field for the stock ticker symbol with default value "AAPL"
    ticker_symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
    
    # Radio buttons to select the data source: Historical Data or Real-Time Data
    data_source = st.radio("Select Data Source:", options=["Historical Data", "Real-Time Data"])
    
    # Flag to indicate if custom dates are used for historical data
    custom_date = False
    if data_source == "Historical Data":
        # Provide options for historical data period selection
        period_option = st.selectbox("Select Historical Data Period:", options=["1y", "6mo", "3mo", "1mo", "Custom"], index=0)
        if period_option == "Custom":
            custom_date = True
            # Date inputs for custom date range
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
    else:
        # For real-time data, allow selection of period and interval
        realtime_period = st.selectbox("Select Real-Time Data Period:", options=["1d", "5d"], index=0)
        realtime_interval = st.selectbox("Select Real-Time Data Interval:", options=["1m", "5m", "15m"], index=0)
    
    st.markdown("---")
    st.subheader("VPT Customization Options")
    # Input for calculation period for the VPT indicator (in days)
    calc_period = st.number_input("VPT Calculation Period (in days):", min_value=1, value=1, step=1)
    # Input for weighting factor to adjust the influence of volume
    weighting_factor = st.number_input("Weighting Factor:", min_value=0.0, value=1.0, step=0.1)
    # Checkbox to decide if smoothing should be applied to the VPT values
    apply_smoothing = st.checkbox("Apply Smoothing to VPT")
    # Input for smoothing window if smoothing is applied
    smoothing_window = st.number_input("Smoothing Window (in days):", min_value=1, value=5, step=1) if apply_smoothing else 5
    
    st.markdown("---")
    st.subheader("Chart Customization Options")
    # Checkbox to toggle chart display
    display_chart = st.checkbox("Display VPT Chart", value=True)
    # Color picker for line color in the chart
    line_color = st.color_picker("Select Line Color", value="#0000FF")
    # Slider to adjust the thickness of the line in the chart
    line_thickness = st.slider("Line Thickness", min_value=1, max_value=10, value=2)
    # Option to toggle grid display on the chart
    show_grid = st.checkbox("Show Grid", value=True)
    # Dropdown to select the style of the line
    line_style = st.selectbox("Select Line Style:", options=["solid", "dashed", "dotted", "dashdot"], index=0)
    # Option to display markers on the chart points
    show_markers = st.checkbox("Show Markers", value=False)
    # Slider to set marker size if markers are enabled
    marker_size = st.slider("Marker Size", min_value=1, max_value=10, value=4) if show_markers else 0
    # Sliders to adjust the chart dimensions
    fig_width = st.slider("Chart Width (inches)", min_value=5, max_value=20, value=10)
    fig_height = st.slider("Chart Height (inches)", min_value=3, max_value=15, value=6)

    # Button to trigger VPT calculation
    if st.button("Calculate VPT"):
        # Fetch stock data based on the selected data source and input parameters
        if data_source == "Historical Data":
            if custom_date:
                data = fetch_historical_data(ticker_symbol, start_date=start_date, end_date=end_date)
            else:
                data = fetch_historical_data(ticker_symbol, period=period_option)
        else:
            data = fetch_realtime_data(ticker_symbol, period=realtime_period, interval=realtime_interval)
        
        if data is not None:
            # Display the last 10 rows of the original stock data for reference
            st.subheader(f"Original Stock Data for {ticker_symbol}")
            st.dataframe(data.tail(10))
            
            # Calculate VPT on the fetched data with the selected parameters
            data_with_vpt = calculate_vpt(
                data,
                calc_period=calc_period,
                weighting_factor=weighting_factor,
                apply_smoothing=apply_smoothing,
                smoothing_window=smoothing_window
            )
            
            # Display the stock data along with the calculated VPT values
            st.subheader("Stock Data with Calculated VPT")
            st.dataframe(data_with_vpt[['date', 'close', 'volume', 'VPT']].tail(20))
            
            # Plot the VPT trend if the display_chart option is selected
            if display_chart:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                # Map the selected line style to matplotlib linestyle
                linestyle_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
                ls = linestyle_map.get(line_style, "-")
                # Plot the VPT line with or without markers based on user preference
                if show_markers:
                    ax.plot(data_with_vpt['date'], data_with_vpt['VPT'], color=line_color,
                            linewidth=line_thickness, linestyle=ls, marker='o', markersize=marker_size)
                else:
                    ax.plot(data_with_vpt['date'], data_with_vpt['VPT'], color=line_color,
                            linewidth=line_thickness, linestyle=ls)
                ax.set_title(f"VPT Trend Over Time for {ticker_symbol}")
                ax.set_xlabel("Date")
                ax.set_ylabel("VPT")
                if show_grid:
                    ax.grid(True)
                st.pyplot(fig)
            
            # Save the calculated data in the Streamlit session state for later use
            st.session_state.vpt_data = data_with_vpt

    # Button to trigger the investment decision process using the VPTAnalysisAgent
    if st.button("Get Investment Decision"):
        # Check if the VPT data has been calculated first
        if "vpt_data" not in st.session_state or st.session_state.vpt_data.empty:
            st.error("Please calculate VPT values first.")
        else:
            # Fetch the current stock price for decision making
            current_price = fetch_current_price(ticker_symbol)
            if current_price is None:
                st.error("Unable to fetch current stock price for decision making.")
            else:
                # Instantiate the VPTAnalysisAgent from the imported module
                vpt_agent_instance = VPTAnalysisAgent()
                # Get the advisor agent instance from the VPTAnalysisAgent
                advisor_agent = vpt_agent_instance.vpt_trading_advisor()
                # Create the decision task with the current VPT data and current stock price
                decision_task = vpt_agent_instance.vpt_analysis(advisor_agent, st.session_state.vpt_data.copy(), current_price)
                # Import Crew from crewai to coordinate the agent and task
                from crewai import Crew
                crew = Crew(agents=[advisor_agent], tasks=[decision_task], verbose=True)
                # Kick off the analysis and get the result
                result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(result)

# Entry point of the Streamlit app
if __name__ == '__main__':
    main()
