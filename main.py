# main.py - Your FastAPI app with dummy product_graph
from fastapi import FastAPI, Request
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from text2sql_agent import outlet_query
from product_api import retriever_tool


app = FastAPI()
query_info = outlet_query()


# messages = [HumanMessage(content="are there any outlets in KL? And what mugs do you sell?")]
# messages = graph.invoke({"messages": messages})
# print(messages)

# class ChatRequest(BaseModel):
#     message: str

# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     data = await request.json()
#     message = data.get("message", "")
    
#     # Get response from chatbot
#     messages = [HumanMessage(content= message)]
#     response = cs_agent.invoke(messages)


#     return {
#         "response": response,
#     }

@app.post("/product")
async def product_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    

    info = retriever_tool(message)


    return {
        "answer": info
    }


@app.post("/outlet")
async def outlet_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # Format the data to match what your LangGraph workflow expects
    messages = [HumanMessage(content=message)]
    
    # Use the pre-compiled workflow instead of creating a new one
    result = query_info.invoke({"messages": messages})
    last_message = result["messages"][-1]  # get the last message

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


