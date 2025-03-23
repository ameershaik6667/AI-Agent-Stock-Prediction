# src/Agents/PortfolioDataAgent.py

import os
from typing import Any, Dict

from crewai import Agent
from openai import OpenAI

# OpenAI API Key (Ensure it's securely stored in environment variables)
client = OpenAI(api_key="")

def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]

    )
    return response.choices[0].message.content

class PortfolioDataAgent(Agent):
    portfolio_data: Dict[str, Any] = {}  # Explicitly define portfolio_data

    def __init__(self):
        super().__init__(
            name="Portfolio Data Agent",
            role="Portfolio Validator",
            goal="Ensure the accuracy and integrity of portfolio data for analysis.",
            backstory=(
                "An AI-driven financial assistant that validates and normalizes portfolio data, "
                "ensuring compatibility with advanced analytical tools for trading and investment insights."
            )
        )

    def fetch_portfolio_data(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch portfolio details securely from user input."""
        self.portfolio_data = {
            "user_id": user_input.get("user_id"),
            "holdings": user_input.get("holdings", [])
        }
        return self.portfolio_data

    def validate_portfolio_data(self) -> (bool, str):
        """Validate portfolio data for accuracy and completeness using GPT."""
        if not self.portfolio_data.get("holdings"):
            return False, "No valid portfolio data found."

        prompt = f"""
        Validate the following portfolio data:
        {self.portfolio_data}

        Ensure each holding has:
        - A valid stock symbol (e.g., AAPL, GOOGL)
        - Quantity as a positive number
        - Purchase price as a positive number

        Respond with 'Valid' if everything is correct, otherwise list issues.
        """
        validation_result = chatgpt_query(prompt)

        if "Valid" in validation_result:
            return True, "Portfolio data is valid."
        return False, validation_result

    def normalize_portfolio_data(self) -> Dict[str, Any]:
        """Normalize portfolio data for compatibility with analysis workflows."""
        normalized_data = []
        for holding in self.portfolio_data.get("holdings", []):
            try:
                normalized_data.append({
                    "symbol": holding["symbol"].upper(),
                    "quantity": float(holding["quantity"]),
                    "purchase_price": float(holding["purchase_price"])
                })
            except (ValueError, KeyError):
                return {"status": "error", "message": "Invalid portfolio data format."}

        self.portfolio_data["holdings"] = normalized_data
        return self.portfolio_data