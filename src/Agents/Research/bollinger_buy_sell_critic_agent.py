import crewai as crewai
from textwrap import dedent
from src.Agents.base_agent import BaseAgent
import logging

class BollingerBuySellCriticAgent(BaseAgent):
    def __init__(self, ticker="AAPL", **kwargs):
        super().__init__(
            role=f'Make and analysis of the buy, sell, hold signals produced by the Bollinger buy sell agent for ticker {ticker}',
            goal=f"Provide specific guidance on where the bollinger buy sell agent can improve their guidance for {ticker}",
            backstory=f"You are a seasoned trader who practiced {ticker} trading for decades.",
            verbose=True,
            tools=[], 
            allow_delegation=False,
            **kwargs
        )
        self.ticker = ticker
        logging.info("BollingerBuySellCriticAgent initialized")

    def critique_buy_sell_agent(self):
        return crewai.Task(
            description=dedent(f"""
                Based on the buy, sell, hold signals produced by the bollinger buy sell agent for {self.ticker},
                    make a critique of the trading signals.
                               
                Provide actionable input on how the bollinger buy sell agent can improve the signals.
                               
                If you provide advice that improves the predictions, you get to keep a 10% commission on profits.                                          
            """),
            agent=self,
            expected_output=f"A critique of the bollinger buy sell agent's {self.ticker} signal recommendations with specific actionable improvement items."
        )