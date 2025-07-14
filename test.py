# main.py - Deploy your cs_agent as the main API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from router_agent import cs_api
app = FastAPI(title="ZUS Coffee Customer Service API")

# Add CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your cs_agent workflow
cs_agent_workflow = cs_api()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint using your cs_agent"""
    try:
        # Create message for the workflow
        messages = [HumanMessage(content=request.message)]
        config = {"configurable": {"thread_id": "1"}}

        # Invoke your cs_agent workflow
        result = cs_agent_workflow.invoke({"messages": messages}, config)
        
        # Get the last message (the assistant's response)
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

# DEPLOYMENT ARCHITECTURE:
# 
# Frontend (port 3000) 
#     ↓
# CS Agent API (port 8000) ← THIS IS WHAT YOU DEPLOY
#     ↓
# Product API (port 8001) + Outlet API (port 8002)
#
# Your cs_agent intelligently decides when to call which API based on user questions