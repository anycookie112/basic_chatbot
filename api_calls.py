from langgraph.prebuilt import create_react_agent
from llm import llm_groq

llm = llm_groq()

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model=llm,  
    tools=[get_weather],  
    prompt="You are a helpful assistant"  
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response)