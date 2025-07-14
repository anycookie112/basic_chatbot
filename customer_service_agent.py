from typing import Literal
from langgraph.graph import START, StateGraph, END
from langgraph.graph import MessagesState
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
import requests
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from utils.llm import *
# llm = ChatOllama(model="qwen2.5:14b")
llm = llm_meta()

API_URL = "http://localhost:8000/outlet"
API = "http://localhost:8000/product"


@tool
def call_product(message: str) -> str:
    """API call to fetch information about the products in ZUS coffee"""
    print("=" * 50)
    print(f"Sending message: {message}")
    endpoint_url = API
    payload = {"message": message}

    try:
        response = requests.post(endpoint_url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("Response JSON:", response_data)
            
            # Return the actual answer to the LLM
            return response_data.get('answer', 'No product information found')
        else:
            error_msg = f"Error: HTTP {response.status_code}"
            print(error_msg)
            return error_msg
            
    except requests.exceptions.JSONDecodeError as e:
        error_msg = f"JSON decode error: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error calling product API: {e}"
        print(error_msg)
        return error_msg

@tool
def call_outlet(message: str) -> str:
    """API call to fetch information about outlet information like store name, address, opening and closing times"""
    print("=" * 50)
    print(f"Sending message: {message}")
    endpoint_url = API_URL
    payload = {"message": message}

    try:
        response = requests.post(endpoint_url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("Response JSON:", response_data)
            
            # Return the actual answer to the LLM
            return response_data.get('answer', 'No outlet information found')
        else:
            error_msg = f"Error: HTTP {response.status_code}"
            print(error_msg)
            return error_msg
            
    except requests.exceptions.JSONDecodeError as e:
        error_msg = f"JSON decode error: {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error calling outlet API: {e}"
        print(error_msg)
        return error_msg


@tool
def calculator(
    a: int,
    b: int,
    operation: Literal["add", "multiply", "divide"]
) -> float:
    """Perform a basic arithmetic operation (add, multiply, divide) on two integers.

    Args:
        a: The first integer.
        b: The second integer.
        operation: The operation to perform ("add", "multiply", or "divide").

    Returns:
        The result of the arithmetic operation.
    """
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    else:
        raise ValueError(f"Unsupported operation: {operation}")





system_prompt = """
You are a customer service agent for ZUS Coffee. You have access to two API tools:

1. **call_product** - For questions about coffee products, food, mugs, merchandise, prices, ingredients
2. **call_outlet** - For questions about store locations, addresses, opening hours, contact information
3. **calculator** - For math related questions which includes, add, multiply and divide.

CRITICAL TOOL SELECTION RULES:
- ALWAYS analyze the current user question independently
- Do NOT let previous tool usage influence current tool selection
- Each question should be evaluated fresh for tool selection
- If user asks about products/coffee/mugs → use call_product
- If user asks about stores/locations/outlets → use call_outlet
- You can switch between tools freely based on the current question

MEMORY USAGE:
- Remember previous conversation context for natural responses
- But choose tools based only on the current question content
- Previous tool usage should not bias current tool selection

Be helpful and maintain conversation flow while selecting the correct tool for each question.
Information from the api calls need to be summarized or packaged for readibility

IMPORTANT:
-You must not make up any information, only use the information provided to you
-If not information is given to give a accurate answer, give an answer based on assumtion with the information given to you, then ask the user to clarify details for an accurate answer
"""


class State(MessagesState):
    query: str = ""
    result: str = ""
    answer: str = ""




tools = [call_product, call_outlet, calculator]

llm_with_tools = llm.bind_tools(tools)

def cs_agent(state: State):
    """Main assistant node that processes user messages and calls tools as needed."""
    return {"messages": [llm_with_tools.invoke([system_prompt] + state["messages"])]}


workflow = StateGraph(State)

workflow.add_node("cs_agent", cs_agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "cs_agent")
workflow.add_conditional_edges(
    "cs_agent",
    tools_condition,
)

workflow.add_edge("tools", "cs_agent")

from langgraph.checkpoint.memory import MemorySaver



memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
graph = workflow.compile()


from langchain_core.messages import HumanMessage, SystemMessage
def interactive_test():    
    print("ZUS Coffee Chatbot - Interactive Test")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    thread_id = "main_conversation"  
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            reset_message = SystemMessage(content="""
            Analyze this new user question independently. 
            Choose the appropriate tool based ONLY on this current question:
            - call_product: for product/coffee/mug/food questions
            - call_outlet: for store/location/address/hours questions
            Ignore previous tool usage patterns.
            """)
            
            messages = [reset_message, HumanMessage(content=user_input)]
            result = graph.invoke({"messages": messages}, config)
            last_message = result["messages"][-1]
            
            print(f"Bot: {last_message.content}")
            
        except Exception as e:
            print(f"Error: {e}")



# if __name__ == "__main__":
#     interactive_test()



def cs_agent ():
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# messages = [HumanMessage(content="are there any outlets in KL? And what mugs do you sell?")]
# messages = graph.invoke({"messages": messages})
# print(messages)

# from utils.show_graph import show_mermaid
# show_mermaid(graph)
