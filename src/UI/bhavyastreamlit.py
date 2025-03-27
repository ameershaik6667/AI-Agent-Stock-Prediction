import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from crewai import Agent
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from Agents.binput import process_user_input  # Import Week 1 function
from Agents.bvisualisation import visualize_portfolio_impact


# Function to interpret the time frame for Yahoo Finance
def get_time_frame(query_time_frame):
    try:
        today = datetime.today()
        
        if "end of this month" in query_time_frame.lower():
            end_of_month = today.replace(day=28) + timedelta(days=4)  # Approx last day
        elif "next year" in query_time_frame.lower():
            end_of_month = today.replace(year=today.year - 1)
        elif "6 months" in query_time_frame.lower():
            end_of_month = today + timedelta(days=-180)
        else:
            end_of_month = today + timedelta(days=30)  # Default 30 days

        last_month = today + timedelta(days=-30)
        end_of_month = today

        # Convert datetime to string format required by Yahoo Finance
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
            return {"error": "Stock data not available for given time frame."}

        # Prepare training data
        stock_data = stock_data[['Close']].dropna()  # Keep only the Close price
        stock_data['Days'] = np.arange(len(stock_data))  # Convert dates to numerical values

        # Feature Scaling
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(stock_data[['Close']])
        stock_data['Scaled_Close'] = scaled_prices

        # Train Linear Regression Model
        X = stock_data[['Days']]
        y = stock_data['Scaled_Close']
        model = LinearRegression()
        model.fit(X, y)

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

        return {
            "stock_symbol": stock_symbol,
            "last_price": float(last_price),
            "predicted_price": float(predicted_price),
            "adjusted_predicted_price": float(predicted_price_adjusted),
            "initial_market_value": float(initial_market_value),
            "final_market_value": float(final_market_value),
        }
        print("stock_symbol= ",stock_symbol)
        print("last_price= ",last_price)
        print("predicted_price= ",predicted_price)
        print("adjusted_predicted_price= ",predicted_price_adjusted)
        print("initial_market_value= ",initial_market_value)
        print("final_market_value= ",final_market_value)

    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return {"error": str(e)}

# # CrewAI Agents
# input_agent = Agent(
#     role="Input Extraction Agent",
#     goal="Extract relevant stock market details from user input",
#     backstory="A specialist in understanding financial queries and extracting key stock market details.",
#     verbose=True,
#     allow_delegation=False
# )

# time_frame_agent = Agent(
#     role="Time Frame Processing Agent",
#     goal="Convert user-defined time frames to a Yahoo Finance-compatible format",
#     backstory="An expert in time series analysis and financial data formatting.",
#     verbose=True
# )

# stock_data_agent = Agent(
#     role="Stock Data Fetcher",
#     goal="Fetch stock market data and calculate market values",
#     backstory="A data-driven analyst skilled in retrieving and analyzing stock market trends.",
#     verbose=True
# )

# output_agent = Agent(
#     role="Output Display Agent",
#     goal="Present the final prediction in a structured format",
#     backstory="A clear and concise communicator, ensuring predictions are easy to understand.",
#     verbose=True
# )


# Define CrewAI Tasks
def run_stock_prediction(user_query):
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

    if not stock_symbol or not time_frame:# if not stock or time frame it will show below error message
        return {"error": "Incomplete data for stock prediction."}

    start_date, end_date = get_time_frame(time_frame)

    print("\n *********************")
    print(f"start_date= {start_date}   end_date= {end_date}")

    if not (start_date and end_date):
        return {"error": "Invalid time frame format."}

    prediction_result = get_stock_prediction(stock_symbol, percentage_change, start_date, end_date)

    if not prediction_result:
        return {"error": "Stock prediction failed."}

    # Debugging print statement
    print("🔍 Prediction Result:", prediction_result)  

    if "error" not in prediction_result:
        visualize_portfolio_impact(prediction_result)

    return prediction_result


# Streamlit UI
def main():
    import yfinance as yf

    stock_data = yf.download("AAPL")# with help of yohu we are downloading aaple stcok

    print(stock_data)
    st.title("📈 Stock Market Prediction")
    st.write("Enter your stock market prediction query and get future projections.")

    user_query = st.text_input(
        "Enter your market prediction query (e.g., 'What will be the market value of AAPL if it decreases by 5% by the end of this month?')"
    )

    if st.button("Predict"):
        with st.spinner("Processing..."):
            result = run_stock_prediction(user_query)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Prediction Successful!")
            st.write("### 📊 Prediction Results")
            st.json(result) # this will display in json formet in browswe

            # Display key information
            st.write(f"**Stock Symbol:** {result['stock_symbol']}")
            st.write(f"**Last Closing Price:** ${result['last_price']:.2f}")
            st.write(f"**Predicted Price:** ${result['predicted_price']:.2f}")
            st.write(f"**Initial Market Value:** ${result['initial_market_value']:.2f}")
            st.write(f"**Final Market Value:** ${result['final_market_value']:.2f}")

            # Convert to DataFrame for chart plotting
            chart_data = pd.DataFrame(# line chart to display
                {
                    "Time": ["Current", "Predicted"],
                    "Stock Price": [result["last_price"], result["predicted_price"]],
                }
            )

            st.line_chart(chart_data.set_index("Time"))

if __name__ == "__main__":
    main()
