from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import BaseMessage
from typing import List
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph, END
from langgraph.graph import MessagesState
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
import requests
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from utils.llm import llm_groq
# llm = ChatOllama(model="qwen2.5:14b")
llm = llm_groq()

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



system_prompt = """
You are a customer service agent. You have access to two API tools that helps you get information to answer customer questions.
How to use the API
you can directly input the user question to the API call and a str argument
or you can rewrite the user's question slightly to get a better query if the query is not good enough in terms of wording.

For outlet information, you will receive a summary of information that is requested by the user when you call the API


"""


class State(MessagesState):
    # Note: MessagesState already has messages as List[BaseMessage], don't override it
    query: str = ""
    result: str = ""
    answer: str = ""




tools = [call_product, call_outlet]

llm_with_tools = llm.bind_tools(tools)

# Node function
def cs_agent(state: State):
    """Main assistant node that processes user messages and calls tools as needed."""
    return {"messages": [llm_with_tools.invoke([system_prompt] + state["messages"])]}


workflow = StateGraph(State)

workflow.add_node("cs_agent", cs_agent)
workflow.add_node("tools", ToolNode(tools))


# Add edges
workflow.add_edge(START, "cs_agent")
workflow.add_conditional_edges(
    "cs_agent",
    tools_condition,
)

workflow.add_edge("tools", "cs_agent")

from langgraph.checkpoint.memory import MemorySaver

def cs_api ():
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# messages = [HumanMessage(content="are there any outlets in KL? And what mugs do you sell?")]
# messages = graph.invoke({"messages": messages})
# print(messages)

# from utils.show_graph import show_mermaid
# show_mermaid(graph)
