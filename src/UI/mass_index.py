import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from yahooquery import Ticker  # For fetching current stock price

# -------------------------------------------
# CrewAI and OpenAI imports
# -------------------------------------------
from crewai import Agent, Task, Crew
from textwrap import dedent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize the OpenAI GPT model for CrewAI agents
gpt_model = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-4o"
)

# -------------------------------------------
# Function to fetch historical or real-time stock data using yfinance
# -------------------------------------------
def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch stock data for the given ticker symbol using yfinance.
    
    Parameters:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        period: Data period to fetch (e.g., '1y' or '1d').
        interval: Data interval (e.g., '1d' for historical, '1m' for real-time).
    
    Returns:
        DataFrame containing the stock data.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error("No data found. Please check the ticker symbol and parameters.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -------------------------------------------
# Helper function to flatten MultiIndex columns (if present)
# -------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns in the DataFrame, if they exist.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(i) for i in col]).strip() for col in df.columns.values]
    return df

# -------------------------------------------
# Helper function to standardize column names and remove common trailing tokens
# -------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase, remove extra whitespace, and if all columns share
    a common trailing token, remove that trailing token.
    """
    df.columns = df.columns.str.lower().str.strip()
    cols = df.columns.tolist()
    split_cols = [col.split() for col in cols]
    if all(len(tokens) >= 2 for tokens in split_cols):
        last_tokens = [tokens[-1] for tokens in split_cols]
        if len(set(last_tokens)) == 1:
            new_cols = [' '.join(tokens[:-1]) for tokens in split_cols]
            df.columns = new_cols
    return df

# -------------------------------------------
# Function to calculate the Mass Index indicator
# -------------------------------------------
def calculate_mass_index(data: pd.DataFrame, ema_period: int = 9, sum_period: int = 25) -> pd.Series:
    """
    Calculate the Mass Index indicator using the daily range (high - low).

    Parameters:
        data: DataFrame containing at least 'high' and 'low' columns.
        ema_period: Period for calculating the exponential moving averages.
        sum_period: Look-back period over which to sum the EMA ratio.
    
    Returns:
        A Pandas Series representing the Mass Index.
    """
    # Flatten and standardize column names
    data = flatten_columns(data)
    data = standardize_columns(data)
    
    required_cols = {"high", "low"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"Data must contain columns: {required_cols}. Available columns: {list(data.columns)}")
        return pd.Series(dtype=float)
    
    # Calculate the daily price range
    price_range = data['high'] - data['low']
    
    # Calculate the exponential moving averages (EMA)
    ema_range = price_range.ewm(span=ema_period, adjust=False).mean()
    ema_ema_range = ema_range.ewm(span=ema_period, adjust=False).mean()
    
    # Compute the ratio of the two EMAs
    ratio = ema_range / ema_ema_range
    
    # Calculate the Mass Index as the rolling sum of the ratio
    mass_index = ratio.rolling(window=sum_period).sum()
    
    return mass_index

# -------------------------------------------
# Function to fetch the current stock price using yahooquery
# -------------------------------------------
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
        if price_data and symbol in price_data and 'regularMarketPrice' in price_data[symbol]:
            return price_data[symbol]['regularMarketPrice']
        else:
            st.error("Failed to fetch current price from ticker.price.")
            return None
    except Exception as e:
        st.error(f"Error fetching current price: {e}")
        return None

# -------------------------------------------
# CrewAI Agent Code for Mass Index Investment Decision
# -------------------------------------------
class MassIndexAnalysisAgents:
    def mass_index_investment_advisor(self):
        """
        Returns an agent that analyzes Mass Index data and current stock price to provide actionable investment recommendations.
        """
        return Agent(
            llm=gpt_model,
            role="Mass Index Investment Advisor",
            goal="Provide actionable investment recommendations (buy, sell, or hold) based on Mass Index data and current stock price.",
            backstory=("You are an experienced technical analyst specializing in volatility indicators. "
                       "Analyze the latest Mass Index value alongside the current stock price to provide clear buy, sell, or hold signals."),
            verbose=True,
            tools=[]  # Additional tools can be added as needed
        )

    def mass_index_analysis(self, agent, mass_index_value, current_price):
        """
        Create a task for the agent to analyze the Mass Index value and current stock price, then provide an investment recommendation.

        Parameters:
            agent: The CrewAI agent instance.
            mass_index_value: The latest Mass Index value.
            current_price: The current stock price.
        
        Returns:
            A Task object for the agent.
        """
        description = dedent(f"""
            Analyze the following data:
            - Mass Index: {mass_index_value}
            - Current Stock Price: {current_price}

            Based on these values and the current market conditions, provide a clear investment recommendation.
            Your final answer should clearly indicate whether to BUY, SELL, or HOLD, along with supporting reasoning.
        """)
        return Task(
            description=description,
            agent=agent,
            expected_output="A detailed analysis and a clear buy, sell, or hold recommendation based on the Mass Index and current stock price."
        )

# -------------------------------------------
# Main Streamlit UI Code with Data Integration and CrewAI Investment Decision
# -------------------------------------------
def main():
    st.title("Stock Data and Mass Index Calculator")
    
    # Sidebar for general configuration
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    
    # Data Mode selection: Historical vs Real-Time
    data_mode = st.sidebar.radio("Select Data Mode", options=["Historical", "Real-Time"])
    
    # Data parameters based on selected mode
    if data_mode == "Historical":
        period_str = st.sidebar.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
        interval = st.sidebar.selectbox("Data Interval", options=["1d", "1wk", "1mo"], index=0)
    else:
        period_str = st.sidebar.selectbox("Real-Time Data Period", options=["1d", "5d"], index=0)
        interval = st.sidebar.selectbox("Real-Time Data Interval", options=["1m", "2m", "5m", "15m"], index=0)
    
    # Sidebar for Mass Index parameters
    st.sidebar.subheader("Mass Index Parameters")
    ema_period = st.sidebar.number_input("EMA Period", min_value=5, max_value=20, value=9, step=1)
    sum_period = st.sidebar.number_input("Sum Period", min_value=10, max_value=50, value=25, step=1)
    
    # Sidebar for Chart Customization
    st.sidebar.subheader("Chart Customization")
    line_color = st.sidebar.color_picker("Mass Index Line Color", value="#0000FF")
    line_style_choice = st.sidebar.selectbox("Mass Index Line Style", options=["solid", "dashed", "dotted", "dashdot"], index=0)
    line_style_map = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
    line_style = line_style_map[line_style_choice]
    chart_width = st.sidebar.slider("Chart Width (inches)", min_value=5, max_value=20, value=10)
    chart_height = st.sidebar.slider("Chart Height (inches)", min_value=3, max_value=15, value=4)
    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    
    # Sidebar for Thresholds and Additional Options
    st.sidebar.subheader("Thresholds and Additional Options")
    show_thresholds = st.sidebar.checkbox("Show Threshold Lines", value=False)
    if show_thresholds:
        upper_threshold = st.sidebar.number_input("Upper Threshold", value=27.0)
        lower_threshold = st.sidebar.number_input("Lower Threshold", value=26.5)
    else:
        upper_threshold = None
        lower_threshold = None
    show_price_chart = st.sidebar.checkbox("Show Closing Price Chart", value=False)
    show_raw_table = st.sidebar.checkbox("Show Raw Stock Data Table", value=True)
    show_data_with_mi_table = st.sidebar.checkbox("Show Data with Mass Index Table", value=True)
    
    # Button to fetch data
    if st.sidebar.button("Fetch Data"):
        st.info(f"Fetching {data_mode} data for {ticker}...")
        stock_data = fetch_stock_data(ticker, period=period_str, interval=interval)
        if not stock_data.empty:
            st.subheader("Fetched Stock Data")
            if show_raw_table:
                st.dataframe(stock_data.tail(10))
            # Store the fetched data in session state for later use
            st.session_state['stock_data'] = stock_data
        else:
            st.error("Failed to fetch data. Please check the ticker symbol and parameters.")
    
    # Button to calculate and display Mass Index
    if st.sidebar.button("Calculate Mass Index"):
        if 'stock_data' not in st.session_state:
            st.error("Please fetch the stock data first.")
        else:
            stock_data = st.session_state['stock_data']
            st.info("Calculating Mass Index...")
            mass_index_series = calculate_mass_index(stock_data, ema_period=ema_period, sum_period=sum_period)
            
            # Append the Mass Index to the data for visualization
            stock_data_with_mi = stock_data.copy()
            stock_data_with_mi['mass index'] = mass_index_series
            
            # Store the Mass Index series in session state for CrewAI use
            st.session_state['mass_index_series'] = mass_index_series
            st.session_state['stock_data_with_mi'] = stock_data_with_mi
            
            # Plot the Mass Index Chart
            st.subheader("Mass Index Chart")
            fig, ax = plt.subplots(figsize=(chart_width, chart_height))
            ax.plot(stock_data_with_mi.index, stock_data_with_mi['mass index'],
                    label='Mass Index', color=line_color, linestyle=line_style)
            ax.set_title(f"{ticker} Mass Index (EMA Period: {ema_period}, Sum Period: {sum_period})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Mass Index")
            if show_grid:
                ax.grid(True)
            if show_thresholds and upper_threshold is not None and lower_threshold is not None:
                ax.axhline(upper_threshold, color='red', linestyle='--', label=f'Upper Threshold ({upper_threshold})')
                ax.axhline(lower_threshold, color='green', linestyle='--', label=f'Lower Threshold ({lower_threshold})')
            ax.legend()
            st.pyplot(fig)
            
            # Optionally plot the Closing Price Chart
            if show_price_chart:
                st.subheader("Closing Price Chart")
                fig2, ax2 = plt.subplots(figsize=(chart_width, chart_height))
                if 'close' in stock_data.columns:
                    ax2.plot(stock_data.index, stock_data['close'], label='Close Price', color='orange')
                    ax2.set_title(f"{ticker} Closing Price")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Price")
                    if show_grid:
                        ax2.grid(True)
                    ax2.legend()
                    st.pyplot(fig2)
                else:
                    st.error("Closing price data not available.")
            
            # Show Data Tables if selected
            if show_data_with_mi_table:
                st.subheader("Data with Mass Index")
                st.dataframe(stock_data_with_mi.tail(10))
            
            # Provide a download button for CSV export
            csv_data = stock_data_with_mi.to_csv().encode('utf-8')
            st.download_button(label="Download Data as CSV",
                               data=csv_data,
                               file_name=f"{ticker}_mass_index.csv",
                               mime='text/csv')
    
    # -------------------------------------------
    # Button to Get Investment Decision using CrewAI for Mass Index
    # -------------------------------------------
    if st.sidebar.button("Get Investment Decision for Mass Index"):
        # Check if the Mass Index data is available in session state
        if 'mass_index_series' not in st.session_state:
            st.error("Please calculate the Mass Index first.")
        else:
            mass_index_series = st.session_state['mass_index_series']
            # Get the latest non-NaN Mass Index value
            try:
                latest_mass_index = mass_index_series.dropna().iloc[-1]
            except Exception as e:
                st.error(f"Error retrieving latest Mass Index value: {e}")
                latest_mass_index = None
            
            # Fetch the current stock price using yahooquery
            current_price = fetch_current_price(ticker)
            
            if latest_mass_index is None or current_price is None:
                st.error("Unable to retrieve necessary data for investment decision.")
            else:
                # Create the CrewAI agent and task for Mass Index analysis
                agents = MassIndexAnalysisAgents()
                advisor_agent = agents.mass_index_investment_advisor()
                analysis_task = agents.mass_index_analysis(advisor_agent, latest_mass_index, current_price)
                
                # Initialize Crew with the agent and task
                crew = Crew(
                    agents=[advisor_agent],
                    tasks=[analysis_task],
                    verbose=True
                )
                
                # Kickoff the CrewAI process to get the investment decision
                result = crew.kickoff()
                st.subheader("Investment Decision")
                st.write(result)

if __name__ == "__main__":
    main()
