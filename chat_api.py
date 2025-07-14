# main.py - Deploy your cs_agent as the main API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from customer_service_agent import cs_agent
app = FastAPI(title="ZUS Coffee Customer Service API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cs_agent_workflow = cs_agent()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint using your cs_agent"""
    try:
        messages = [HumanMessage(content=request.message)]
        config = {"configurable": {"thread_id": "1"}}

        result = cs_agent_workflow.invoke({"messages": messages}, config)
        
        last_message = result["messages"][-1]
        
        return {
            "response": last_message.content,
            "user_message": request.message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ZUS Coffee CS Agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

