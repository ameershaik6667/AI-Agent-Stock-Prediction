
from crewai import Agent, Crew, Task

from src.Agents.bagents import (analyze_market_data, extract_stock_symbol,
                                fetch_market_sentiment, get_stock_data,
                                visualize_stock_data)

# Define Agents
analysis_agent = Agent(
    role="Analysis Agent",
    goal="Analyze stock data and sentiment to provide financial insights.",
    backstory="You are an AI-powered financial strategist. You analyze stock indicators "
              "and market sentiment to provide actionable investment recommendations.",
    function=analyze_market_data
)

# Define Tasks with Expected Outputs
analyze_task = Task(
    description="Analyze stock data and sentiment for investment insights based on {query}.",
    agent=analysis_agent,
    expected_output="A text-based recommendation on whether to buy, hold, or sell the stock based on the analysis."
)

# Create Crew
finance_crew = Crew(agents=[analysis_agent], tasks=[analyze_task])

if __name__ == "__main__":
    print("\nStarting Financial Analysis Crew...\n")
    while True:
        query = input("\nEnter a financial query (or type 'exit' to stop): ").strip()
        if query.lower() == "exit":
            print("\nExiting...")
            break
        
        # Step 1: Extract stock symbol
        parsed_data = extract_stock_symbol(query)
        if not parsed_data:
            print("Error parsing query.")
            continue
        
        symbol = parsed_data.get("symbol")
        if not symbol:
            print("No stock symbol found.")
            continue
        
        # Step 2: Fetch stock data
        stock_data = get_stock_data(symbol)
        if not stock_data:
            print("Error fetching stock data.")
            continue
        
        # Step 3: Fetch market sentiment
        sentiment_data = fetch_market_sentiment(symbol)
        if not sentiment_data:
            print("Error fetching sentiment data.")
            continue
        
        # Step 4: Analyze market conditions
        analysis_result = analyze_market_data(stock_data, sentiment_data)
        
        # Step 5: Visualize stock trends
        visualize_stock_data(symbol, stock_data)
        
        # Run the Crew
        result = finance_crew.kickoff(inputs={"query": analysis_result})
        
        print("\nAnalysis Result:\n", result)
        print("\n" + "-" * 50)
