# main.py - Your FastAPI app with dummy product_graph
from fastapi import FastAPI, Request
import random
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from agents import create_super
from fastapi.responses import JSONResponse
from text2sql_agent import outlet_query



llm = ChatOllama(model="qwen2.5:14b")

app = FastAPI()


@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # Get response from chatbot
    response = llm.invoke(message)
    
    return {
        "response": response,
        "user_message": message
    }

@app.post("/product")
async def product_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    

    supervisor = create_super()

    messages = [HumanMessage(content=message)]
    messages = supervisor.invoke({"messages": messages})
    # print(messages)

    last_message = messages["messages"][-1]  # get the last message
    print(last_message.content)


    return {
        "answer": last_message
    }


@app.post("/outlet")
async def outlet_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # Format the data to match what your LangGraph workflow expects
    messages = [HumanMessage(content=message)]

    
    query_info = outlet_query()
    
    messages = query_info.invoke({"messages": messages})
    last_message = messages["messages"][-1]  # get the last message


    return {
        "answer": last_message
    }

"""

so when main api is triggered
llm will decide where to route 

this llm will be given 2 tools
the llm will decide which api to call for the information needed to answer the user's question
then get the info generate the answer
send the answer to the user



"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


