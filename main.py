# main.py - Your FastAPI app with dummy product_graph
from fastapi import FastAPI, Request
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from api_calls.text2sql_agent import outlet_query
from api_calls.product_api import retriever_tool


app = FastAPI()
query_info = outlet_query()


@app.post("/products")
async def product_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    

    info = retriever_tool(message)


    return {
        "answer": info
    }


@app.post("/outlets")
async def outlet_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    messages = [HumanMessage(content=message)]
    
    result = query_info.invoke({"messages": messages})
    last_message = result["messages"][-1]  

    return {
        "answer": last_message.content
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


