import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from crewai import Agent, Crew, Task
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.Agents.hafunctions import (RNNModel, calculate_indicators,
                                    extract_stock_symbol, forward_test,
                                    generate_prompt, get_stock_data,
                                    prepare_data, train_model)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)



# Define CrewAI Agent for extracting stock symbol
stock_symbol_agent = Agent(
    role="Stock Symbol Extractor",
    goal="Extract a valid stock symbol from the user's input.",
    backstory="An AI-powered assistant that identifies stock symbols from user queries.",
    verbose=True
)
# Define CrewAI Agent for Stock Analysis
stock_analysis_agent = Agent(
    role="Stock Market Analyst",
    goal="Analyze stock market trends and provide trading recommendations.",
    backstory="An AI-powered financial analyst with expertise in technical indicators and stock predictions.",
    verbose=True
)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    while True:
        # Ask user for a stock query
        user_input = input("Enter stock query (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        # Define the Task for extracting stock symbol
        stock_symbol_task = Task(
            description=f"Extract a valid stock symbol from the following user input: '{user_input}'",
            agent=stock_symbol_agent,
            expected_output="A valid stock symbol."
        )
        
        # Create Crew and execute extraction
        crew = Crew(agents=[stock_symbol_agent], tasks=[stock_symbol_task])
        ticker = crew.kickoff()
        print(ticker)
        ticker = str(ticker).upper()
        
        # Load data
        stock_data = get_stock_data(ticker)
        x_data, y_data, scaler = prepare_data(stock_data)

        # Split data into training and validation sets
        train_size = int(0.8 * len(x_data))
        x_train, y_train = x_data[:train_size], y_data[:train_size]
        x_val, y_val = x_data[train_size:], y_data[train_size:]

        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Define and train the model
        model = RNNModel(input_size=x_train.shape[2]).to(device)
        train_model(model, train_loader, val_loader)

        # Run forward test and generate prompt
        predicted_price = forward_test(model, ticker)
        prompt = generate_prompt(ticker, model, scaler)
        print(prompt)

                # Define the Task for the Stock Analysis Agent
        stock_analysis_task = Task(
            description=f"Analyze the following stock data and provide insights and recommendations:\n\n{prompt}",
            agent=stock_analysis_agent,
            expected_output="A concise analysis of the stock with actionable trading recommendations."
        )

        # Execute analysis
        analysis_crew = Crew(agents=[stock_analysis_agent], tasks=[stock_analysis_task])
        analysis_output = analysis_crew.kickoff()

if __name__ == "__main__":
    main()


