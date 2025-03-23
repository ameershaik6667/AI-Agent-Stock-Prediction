# src/Agents/SignalAnalysisAgent.py

import json
from typing import Any, Dict, List

from crewai import Agent
from openai import OpenAI

from Agents.NARindicators import generate_technical_indicators

# OpenAI API Key (Ensure it's securely stored in environment variables)
client = OpenAI(api_key="")


def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


class SignalAnalysisAgent(Agent):
    signals: List[Dict[str, Any]] = []

    def __init__(self):
        super().__init__(
            name="Signal Analysis Agent",
            role="Market Signal Analyzer",
            goal="Analyze financial portfolios and generate actionable trading signals.",
            backstory=(
                "A market-savvy AI specializing in stock analysis, trend detection, and "
                "providing actionable buy, sell, or hold recommendations based on market conditions."
            )
        )

    def analyze_portfolio(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolio data to generate trading signals with AI indicators.
        """
        if not portfolio_data or not portfolio_data.get("holdings"):
            return {"status": "error", "message": "No valid portfolio data provided."}

        enhanced_holdings = []
        for holding in portfolio_data["holdings"]:
            symbol = holding["symbol"]

            # Fetch AI-based indicators (RSI, MACD, moving averages)
            indicators = generate_technical_indicators(symbol)
            if indicators:
                holding.update(indicators)
            enhanced_holdings.append(holding)

        # Prepare prompt for GPT-based signal analysis
        prompt = f"""
        Analyze the following stock portfolio and generate trading signals:
        
        Portfolio Holdings:
        {json.dumps(enhanced_holdings, indent=2)}

        For each stock, consider:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Signal Line
        - SMA50 (50-day Simple Moving Average)
        - SMA200 (200-day Simple Moving Average)
        - Current price trends

        Provide a trading signal (Buy, Sell, or Hold) with a short explanation.

        Respond in JSON format like this:
        [
            {{"symbol": "AAPL", "signal": "Buy", "reason": "RSI below 30 and MACD above Signal Line."}},
            {{"symbol": "GOOGL", "signal": "Hold", "reason": "Stable trend, no strong buy/sell signals."}}
        ]
        """
        analysis_result = chatgpt_query(prompt)

        # Parse GPT response into structured signals
        try:
            parsed_result = json.loads(analysis_result)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Failed to parse AI response."}

        # Store and return the analysis result
        self.signals = parsed_result
        return {"status": "success", "signals": self.signals}
