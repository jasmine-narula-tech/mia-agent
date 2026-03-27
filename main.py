from google.adk.agents import Agent
from google.adk.tools import google_search # Example tool

# Define the specialized agent
mia_agent = Agent(
    name="MIA",
    model="gemini-3-flash", # Use the 2026 frontier model
    instruction="""
    You are a professional meeting analyst. 
    1. Extract action items.
    2. Identify blockers.
    3. Calculate speaker engagement.
    Return ONLY JSON.
    """,
    # You can add tools here so the agent can actually DO things
    tools=[google_search] 
)