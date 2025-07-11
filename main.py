# # main.py - Your FastAPI app with dummy product_graph
# from fastapi import FastAPI, Request
# import random
# from llm import llm_groq

# llm =llm_groq()

# app = FastAPI()

# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     data = await request.json()
#     message = data.get("message", "")
    
#     # Get response from chatbot
#     response = llm.invoke(message)
    
#     return {
#         "response": response,
#         "user_message": message
#     }

# # Health check endpoint
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "message": "FastAPI server is running"}

# # Root endpoint
# @app.get("/")
# async def root():
#     return {"message": "ZUS Coffee Chatbot API", "endpoints": ["/chat", "/health"]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py - FastAPI app with streaming support
# main.py - FastAPI app with streaming support
from test import create_super
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
from llm import llm_groq

llm = create_super()

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    # Handle both formats
    if "messages" in data:
        # OpenAI-style format
        messages = data["messages"]
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        message = user_message or ""
    else:
        # Your original format
        message = data.get("message", "")
    
    # Get response from chatbot (non-streaming)
    response = llm.invoke(message)
    
    return {
        "response": response,
        "user_message": message
    }

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    # Handle both formats
    if "messages" in data:
        # OpenAI-style format
        messages = data["messages"]
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        message = user_message or ""
    else:
        # Your original format
        message = data.get("message", "")
    
    async def generate_response():
        try:
            if hasattr(llm, 'stream'):
                #Generates the LLM response in chunks
                for chunk in llm.stream(message):
                    content = getattr(chunk, 'content', '')
                    
                    if content:
                        chunk_data = {
                            "type": "chunk", 
                            "content": content,
                            "user_message": message
                        }
                        #Sends data piece-by-piece from a generator
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.01)
                    
            elif hasattr(llm, 'astream'):
                async for chunk in llm.astream(message):
                    content = getattr(chunk, 'content', '')
                    
                    # Only send non-empty 
                    if content:
                        chunk_data = {
                            "type": "chunk",
                            "content": content,
                            "user_message": message
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
            else:
                # Fallback: simulate streaming by breaking response into chunks
                full_response = llm.invoke(message)
                words = full_response.split()
                
                for i, word in enumerate(words):
                    chunk_data = {
                        "type": "chunk",
                        "content": word + " ",
                        "user_message": message,
                        "is_last": i == len(words) - 1
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.05)  # Simulate typing delay
            
            # Send completion signal
            completion_data = {
                "type": "complete",
                "message": "Stream completed"
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "FastAPI server is running"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ZUS Coffee Chatbot API", 
        "endpoints": ["/chat", "/chat/stream", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)