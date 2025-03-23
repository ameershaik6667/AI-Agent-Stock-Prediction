from typing import Any, Dict

from crewai import Agent
from pydantic import PrivateAttr

from Agents.NARindicators import generate_technical_indicators

from .NARPortfolioDataAgent import PortfolioDataAgent
from .NARSignalAnalysisAgent import SignalAnalysisAgent


class ScenarioInputAgent(Agent):
    _portfolio_data_agent: PortfolioDataAgent = PrivateAttr()
    _signal_analysis_agent: SignalAnalysisAgent = PrivateAttr()

    def __init__(self, portfolio_data_agent: PortfolioDataAgent, signal_analysis_agent: SignalAnalysisAgent, **kwargs):
        super().__init__(
            name="Scenario Input Agent",
            role="Query Analyzer",
            goal="Analyze user queries and route them to appropriate agents for processing.",
            backstory=(
                "An AI-driven assistant specializing in understanding financial queries and "
                "directing them to the best-suited analytical tools for precise decision-making."
            ),
            **kwargs
        )
        self._portfolio_data_agent = portfolio_data_agent
        self._signal_analysis_agent = signal_analysis_agent

    def execute(self, query: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes user queries, validates input, and routes to appropriate agents.
        """

        # Validate and fetch portfolio data
        portfolio_data = self._portfolio_data_agent.fetch_portfolio_data(user_input)
        is_valid, message = self._portfolio_data_agent.validate_portfolio_data()

        if not is_valid:
            return {"status": "error", "message": message}

        normalized_data = self._portfolio_data_agent.normalize_portfolio_data()

        # Enhance each holding with AI-based indicators
        for holding in normalized_data["holdings"]:
            symbol = holding["symbol"]
            indicators = generate_technical_indicators(symbol)

            if indicators:
                holding.update(indicators)

        # Scenario-based analysis using AI indicators
        prompt = f"""
        Analyze the impact of the following financial scenario on the user's portfolio:
        
        Scenario Description:
        {query}

        Portfolio Holdings (with technical indicators):
        {normalized_data}

        Consider the following:
        - RSI (Relative Strength Index) trends
        - MACD and Signal Line crossovers
        - Short-term (SMA50) vs. long-term (SMA200) moving averages
        - How external factors might impact stock performance

        Provide a detailed impact analysis and suggest investment actions.
        """
        
        # Use AI to generate the analysis
        scenario_analysis = self._signal_analysis_agent.chatgpt_query(prompt)

        return {
            "status": "success",
            "task": "scenario_analysis",
            "analysis": scenario_analysis
        }
