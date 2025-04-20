import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from crewai import Agent, Crew, Task
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from Agents.bhbacktest import run_backtest

# Suppress all warnings
warnings.filterwarnings("ignore")

from Agents.binput import process_user_input


# Function to interpret time frames
def get_time_frame(query_time_frame):
    try:
        # Make sure we're working with a string
        if not isinstance(query_time_frame, str):
            query_time_frame = str(query_time_frame)
            
        today = datetime.today()
        if "end of this month" in query_time_frame.lower():
            end_of_month = today.replace(day=28) + timedelta(days=4)  # Approx last day
        elif "next year" in query_time_frame.lower():
            end_of_month = today.replace(year=today.year + 1)
        elif "6 months" in query_time_frame.lower():
            end_of_month = today + timedelta(days=180)
        else:
            end_of_month = today + timedelta(days=30)  # Default 30 days

        last_month = today - timedelta(days=30)
        return last_month.strftime('%Y-%m-%d'), end_of_month.strftime('%Y-%m-%d')

    except Exception as e:
        print(f"Error processing time frame: {e}")
        return None, None

# Function to fetch stock data and apply ML prediction
def get_stock_prediction(stock_symbol, percentage_change, start_date, end_date):
    try:
        # Download past 1 year of stock data for training
        today = datetime.today()
        one_year_ago = today - timedelta(days=365)
        stock_data = yf.download(stock_symbol, start=one_year_ago.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

        if stock_data.empty:
            return {"error": "Stock data not available for the given time frame."}

        # Prepare training data
        stock_data = stock_data[['Close']].dropna()
        stock_data['Days'] = np.arange(len(stock_data))

        # Feature Scaling
        scaler = MinMaxScaler()
        stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(stock_data[['Days']], stock_data['Scaled_Close'])

        # Predict Future Price
        future_days = len(stock_data) + 30  # Predict 30 days into the future
        scaled_predicted_price = model.predict([[future_days]])[0]
        predicted_price = scaler.inverse_transform([[scaled_predicted_price]])[0][0]

        # Current price & Market Value Calculation
        last_price = stock_data['Close'].iloc[-1]
        stock_info = yf.Ticker(stock_symbol).info
        total_shares_outstanding = stock_info.get('sharesOutstanding', 0)

        # Apply Percentage Change (if given)
        predicted_price_adjusted = last_price * (1 + (percentage_change / 100))
       
        # Market Value Calculations
        initial_market_value = last_price * total_shares_outstanding
        final_market_value = predicted_price_adjusted * total_shares_outstanding
        
        # Prepare data for backtesting - use the complete stock data with OHLCV
        full_stock_data = yf.download(stock_symbol, start=one_year_ago.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

        backtest_results = run_backtest(full_stock_data)


        return {
            "stock_symbol": stock_symbol,
            "last_price": float(last_price),
            "predicted_price": float(predicted_price),
            "adjusted_predicted_price": float(predicted_price_adjusted),
            "initial_market_value": float(initial_market_value),
            "final_market_value": float(final_market_value),
            "backtest": backtest_results,

        }

    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return {"error": str(e)}



# CrewAI Agent for Stock Analysis
stock_agent = Agent(
    role="Stock Market Analyst",
    goal="Analyze stock prediction results and provide investment insights",
    backstory="An experienced financial analyst who evaluates market trends and advises investors.",
)

# CrewAI Task for Investment Recommendation
stock_recommendation_task = Task(
    description="Based on the predicted stock price for {stock_symbol}, "
                "provide an investment recommendation. Consider whether the price "
                "is expected to increase or decrease, and offer strategic advice accordingly.",
    agent=stock_agent,
    function=lambda stock_symbol, predicted_price, last_price, final_market_value: (
        f"Stock: {stock_symbol}\n"
        f"Current Price: ${last_price:.2f}\n"
        f"Predicted Price: ${predicted_price:.2f}\n"
        f"Final Market Value Estimate: ${final_market_value:.2f}\n\n"
        f"📢 Recommendation: "
        f"{'Consider buying, as the price is expected to increase.' if predicted_price > last_price else 'Consider selling or holding, as the price may drop.'}"
    ),
    expected_output="A recommendation message with investment advice along with numbers defining it.",
)

# CrewAI Crew
stock_crew = Crew(
    agents=[stock_agent],
    tasks=[stock_recommendation_task],
    process="sequential",
)

# Function to run stock prediction
def run_stock_prediction(user_query):
    """Processes user query, extracts data, and runs stock prediction."""
    extracted_data = process_user_input(user_query)

    if not extracted_data:
        return {"error": "Failed to extract stock details."}

    stock_symbol = extracted_data.get("Stock Symbol")
    percentage_change_str = extracted_data.get("Percentage Change", "0")
    time_frame = extracted_data.get("Time Frame")

    try:
        percentage_change = float(percentage_change_str.replace('%', '')) if percentage_change_str else 0.0
    except ValueError:
        return {"error": "Invalid percentage change format."}

    if not stock_symbol or not time_frame:
        return {"error": "Incomplete data for stock prediction."}

    start_date, end_date = get_time_frame(time_frame)

    if not (start_date and end_date):
        return {"error": "Invalid time frame format."}

    # Directly call `get_stock_prediction()`
    prediction_result = get_stock_prediction(stock_symbol, percentage_change, start_date, end_date)
    
    # Generate recommendation using CrewAI
    recommendation_result = stock_crew.kickoff(
        inputs={
            "stock_symbol": prediction_result["stock_symbol"],
            "predicted_price": prediction_result["predicted_price"],
            "last_price": prediction_result["last_price"],
            "final_market_value": prediction_result["final_market_value"],
        }
    )
    
    return prediction_result, recommendation_result


# CLI Interface
def main():
    print(" Stock Market Prediction CLI")
    print("Enter your stock market prediction query and get future projections.")
    print("Example: 'What will be the market value of AAPL if it decreases by 5% by the end of this month?'\n")

    user_query = input("Enter your market prediction query: ")

    print("\n Processing your request...\n")
    result, recommendation = run_stock_prediction(user_query)
    print(result)
    
    print("\nBacktest Results:")
    print(json.dumps(result.get("backtest", {}), indent=4))

    
    print("\n Investment Recommendation:" )
    print(recommendation)

if __name__ == "__main__":
    main()
