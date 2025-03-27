import os

import yfinance as yf
from crewai import Crew, Task
from dotenv import load_dotenv

from Agents.PortfolioDataAgent import PortfolioDataAgent
from Agents.ScenarioInputAgent import ScenarioInputAgent
from Agents.SignalAnalysisAgent import SignalAnalysisAgent
from Agents.TaxRulesAgent import TaxRulesAgent

load_dotenv()

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in the .env file.")

def get_current_price(symbol):
    """Fetch the latest stock price from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]  # Get latest closing price
        return round(price, 2) if not price is None else 0
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return 0  # Default to 0 if fetching fails

def main():
    # Create instances of agents
    portfolio_data_agent = PortfolioDataAgent()
    signal_analysis_agent = SignalAnalysisAgent()
    tax_rules_agent = TaxRulesAgent()
    scenario_input_agent = ScenarioInputAgent(portfolio_data_agent, signal_analysis_agent)

    # Collect user input for portfolio data
    user_input = {
        "user_id": input("Enter your User ID: "),
        "holdings": []
    }

    # Loop to collect holdings data
    while True:
        symbol = input("Enter stock symbol (or 'done' to finish): ").upper()
        if symbol == 'DONE':
            break
        
        # Fetch real-time stock price
        current_price = get_current_price(symbol)
        print(f"Current market price for {symbol}: ${current_price}")

        # Input validation for quantity and purchase price
        while True:
            quantity = input(f"Enter quantity for {symbol}: ")
            if quantity.isdigit() and int(quantity) > 0:
                quantity = float(quantity)
                break
            print("Invalid quantity. Please enter a positive number.")
        
        while True:
            purchase_price = input(f"Enter purchase price for {symbol}: ")
            try:
                purchase_price = float(purchase_price)
                if purchase_price > 0:
                    break
            except ValueError:
                pass
            print("Invalid purchase price. Please enter a positive number.")
        
        # Append holding to the list
        user_input["holdings"].append({
            "symbol": symbol,
            "quantity": quantity,
            "purchase_price": purchase_price,
            "current_price": current_price  # Store real-time price
        })

    while True:  # Loop to allow continuous execution
        # Select Task: Tax Analysis, Signal Generation, or Scenario Analysis
        print("\nSelect Task:")
        print("1. Tax Analysis")
        print("2. Signal Generation")
        print("3. Scenario Analysis")
        task_choice = input("Enter choice (1-3): ")

        if task_choice == "1":
            print("\nSelect Jurisdiction for Tax Calculation:")
            print("1. US")
            print("2. UK")
            print("3. EU")
            print("4. IN")
            jurisdiction_map = {"1": "US", "2": "UK", "3": "EU", "4": "IN"}
            jurisdiction_choice = input("Enter choice (1-4): ")
            jurisdiction = jurisdiction_map.get(jurisdiction_choice, "US")
            response = tax_rules_agent.execute(user_input, jurisdiction)

        else:
            # Define dynamic tasks based on user selection
            tasks = []
            
            if task_choice == "2":  # Generate trading signals
                task_description = f"Analyze the portfolio holdings: {user_input['holdings']} and generate trading signals using RSI, MACD, and moving averages to generate AI-powered signals."
                task = Task(
                    description=task_description,
                    agent=signal_analysis_agent,
                    expected_output="Buy/Sell/Hold signals along with recommendations and reasoning. No Need for rendering design/graphics."
                )
                tasks.append(task)

            elif task_choice == "3":  # Scenario impact analysis
                scenario = input("Describe the financial scenario for analysis: ")
                task_description = f"Analyze the impact of '{scenario}' on the portfolio: {user_input['holdings']} using RSI, MACD, and moving averages to generate AI-powered signals."
                task = Task(
                    description=task_description,
                    agent=scenario_input_agent,
                    expected_output="A detailed impact analysis of the given financial scenario on portfolio performance.Buy/Sell/Hold signals along with recommendations and reasoning. No Need for rendering design/graphics."
                )
                tasks.append(task)

            # Initialize CrewAI to process the tasks dynamically
            crew = Crew(agents=[portfolio_data_agent, signal_analysis_agent, tax_rules_agent, scenario_input_agent], tasks=tasks)
            response = crew.kickoff()

        # Display Response
        print("\nResponse:")
        print(response)

        # Ask user if they want to continue
        continue_choice = input("\nDo you want to perform another task? (yes/no): ").strip().lower()
        if continue_choice != "yes":
            print("Exiting the program. Goodbye!")
            break  # Exit loop if user chooses 'no'

if __name__ == "__main__":
    main()
