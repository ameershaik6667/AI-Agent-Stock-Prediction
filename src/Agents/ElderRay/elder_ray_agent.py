# Import necessary classes and functions for creating and managing agents and tasks
from crewai import Agent, Task
# Import ChatOpenAI to initialize the GPT model for generating responses
from langchain_openai import ChatOpenAI
# Import dedent to clean up multi-line strings by removing unnecessary indentation
from textwrap import dedent
# Import the BaseAgent class to extend common agent functionalities
from src.Agents.base_agent import BaseAgent  # Ensure this base agent exists in your project

# ---------------------------
# Initialize GPT Model
# ---------------------------
# Configure the GPT model with desired parameters. Lower temperature ensures more deterministic outputs.
gpt_model = ChatOpenAI(
    temperature=0,         # Set temperature to 0 for deterministic responses
    model_name="gpt-4o"    # Specify the GPT model version (adjust if necessary)
)

# ---------------------------
# ElderRayAnalysisAgent Class
# ---------------------------
class ElderRayAnalysisAgent(BaseAgent):
    def __init__(self):
        """
        Initialize the ElderRayAnalysisAgent by calling the BaseAgent initializer with:
        - role: The role of this agent.
        - goal: The overall objective the agent should achieve.
        - backstory: Context about the agent's expertise and analytical approach.
        """
        super().__init__(
            role="Elder-Ray Investment Advisor",
            goal="""Interpret Elder-Ray index values along with the current stock price 
                    and provide a clear investment recommendation (BUY, SELL, or HOLD) with supporting reasoning.""",
            backstory="""As an expert technical analyst specializing in the Elder-Ray indicator, 
                         I evaluate buying and selling pressures and integrate current price data 
                         to offer actionable investment decisions."""
        )
    
    def elder_ray_investment_advisor(self):
        """
        Returns an Agent instance configured specifically for providing Elder-Ray investment advice.
        The agent uses the GPT model along with defined role, goal, and backstory.
        """
        return Agent(
            llm=gpt_model,  # Set the language model for this agent
            role="Elder-Ray Investment Advisor",  # Define the agent's role
            goal="""Interpret Elder-Ray index values along with the current stock price 
                    and provide a clear investment recommendation (BUY, SELL, or HOLD) with supporting reasoning.""",
            backstory="""I am a technical analysis expert focusing on the Elder-Ray indicator, 
                         utilizing data-driven insights to guide investment decisions.""",
            verbose=True,  # Enable verbose mode for detailed output/logging
            tools=[]       # No additional tools are provided to the agent
        )
    
    def elder_ray_analysis(self, agent, elder_ray_data, current_price):
        """
        Creates and returns a Task for analyzing the Elder-Ray data along with the current stock price.
        
        Parameters:
        - agent (Agent): The agent that will perform the analysis.
        - elder_ray_data (pd.DataFrame): DataFrame containing the Elder-Ray index values.
        - current_price (float): The current market price of the stock.
        
        Returns:
        - Task: A task instance with a detailed description instructing the agent on what to analyze.
        """
        # Retrieve the last row of Elder-Ray data and convert it to a string (excluding the index)
        last_row = elder_ray_data.tail(1).to_string(index=False)
        
        # Construct a detailed description for the task using dedent to remove extra indentation
        description = dedent(f"""
            Analyze the following Elder-Ray index data:
            {last_row}
            
            Current Stock Price: {current_price}
            
            Based on the above data, please provide an investment recommendation: BUY, SELL, or HOLD.
            Include detailed supporting reasoning.
        """)
        
        # Return a Task instance with the specified description and expected output format
        return Task(
            description=description,
            agent=agent,
            expected_output="A comprehensive investment recommendation (BUY/SELL/HOLD) with detailed reasoning."
        )
