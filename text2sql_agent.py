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
from utils.show_graph import show_mermaid

llm = ChatOllama(model="qwen2.5:14b")
db = SQLDatabase.from_uri("sqlite:///zus_outlets.db")
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
import sqlite3

# print(db.get_table_info())

# Use MessagesState with additional fields
class State(MessagesState):
    # Note: MessagesState already has messages as List[BaseMessage], don't override it
    query: str = ""
    result: str = ""
    answer: str = ""

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database. Use like instead of "=="
due to users not giving the full name of the outlet sometimes.

When joining tables, always include all of the tables

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}


"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: str

@tool
def write_query_tool(question: str) -> str:
    """Generate SQL query to fetch information about ZUS coffee shop outlets."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 15,
            "table_info": db.get_table_info(),
            "input": question,
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return result["query"]

@tool
def execute_query_tool(sql_query: str) -> str:
    """Execute SQL query against the ZUS coffee shop database."""
    # Execute the query
    query_tool = QuerySQLDatabaseTool(db=db)
    try:
        result = query_tool.invoke(sql_query)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def generate_answer_tool(queried_info: str) -> str:
    """Process queried information and generate a helpful answer for the user."""
    try:
        # Create a prompt to help the LLM process the database results
        prompt = f"""
        Based on the following database query results about ZUS coffee shops, 
        provide a helpful, friendly, and informative response to the user. 
        Format the information in a clear and readable way.

        Database Results:
        {queried_info}

        Please provide a natural language response that:
        1. Summarizes the key information
        2. Presents locations in a user-friendly format
        3. Includes relevant details like addresses, contact info, or hours if available
        4. Is conversational and helpful
        """
        
        result = llm.invoke(prompt)
        return result
        # return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Available tools
tools = [write_query_tool, execute_query_tool, generate_answer_tool]

# System message for the assistant
sys_msg = SystemMessage(content="""You are a helpful assistant for ZUS coffee shop information. 
You have access to a database with information about ZUS coffee outlets in Malaysia and each stores operation hours.

When a user asks about ZUS coffee shops, outlets, locations, or stores, follow this workflow:
1. First use write_query_tool to generate an appropriate SQL query
2. Then use execute_query_tool to run that query and get the raw results
3. Finally, use generate_answer_tool to process the raw results into a user-friendly response
Additional: if users also inquiries about the opening time, do also query the information and let the user know about the information they want

Try to gather all the information like address, store name and opening/closing times before generating the answer
so the operation table stores the opening and closing time of the outlet
while the outlet table stores the addreses and the direction url
                                                                                                
If the user asks general questions or greetings, respond naturally and offer to help them find ZUS coffee shop information.
Always reply with something and not keep the users hanging

Do not show any mixups or error messages in the final answer, dont include it in the final message to the users for a better experiece and also dont include, words like in out database, the information gathered 
Be confident in your answer, but to not make up information. Only use information that is presented to you                       
                                                
Always use the tools in sequence for database queries to ensure the best user experience.""")

# LLM with tools bound
llm_with_tools = llm.bind_tools(tools)

# Node function
def assistant(state: State):
    """Main assistant node that processes user messages and calls tools as needed."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from langgraph.graph import StateGraph, END

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode(tools))


# Add edges
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges(
    "assistant",
    tools_condition,
)

workflow.add_edge("tools", "assistant")
# Set entry point

# Compile the graph
def outlet_query():
    return workflow.compile()





# messages = [HumanMessage(content="How many stores are there in total? ")]
# messages = app.invoke({"messages": messages})
# # print(messages)

# last_message = messages["messages"][-1]  # get the last message
# print(last_message.content)