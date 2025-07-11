# # test_client.py - Interactive chatbot test client
# import requests
# import json

# # Base URL (adjust if running on different port)
# BASE_URL = "http://localhost:8000"

# def test_chatbot_interactive():
#     """Interactive chatbot testing"""
#     print("🤖 ZUS Coffee Chatbot Test")
#     print("=" * 40)
#     print("Type 'quit' to exit")
#     print("Try: 'hello', 'show me mugs', 'what bottles do you have?', etc.")
#     print("=" * 40)
    
#     while True:
#         # Get user input
#         user_input = input("\n👤 You: ").strip()
        
#         if user_input.lower() in ['quit', 'exit', 'bye']:
#             print("👋 Goodbye!")
#             break
        
#         if not user_input:
#             continue
        
#         try:
#             # Send message to chatbot
#             payload = {"message": user_input}
#             response = requests.post(
#                 f"{BASE_URL}/chat",
#                 json=payload,
#                 headers={"Content-Type": "application/json"}
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 print(f"🤖 Bot: {result['response']}")
#             else:
#                 print(f"❌ Error: {response.status_code}")
#                 print(f"   Response: {response.text}")
                
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Connection error: {e}")
#             print("Make sure your FastAPI server is running on localhost:8000")

# def test_chatbot_batch():
#     """Test chatbot with predefined messages"""
    
#     test_messages = [
#         "Hello!",
#         "What mugs do you have?",
#         "Show me tumblers",
#         "I need a water bottle",
#         "What about thermal flasks?",
#         "What are your prices?",
#         "Help me find something",
#         "Thanks for your help!"
#     ]
    
#     print("🧪 Testing Chatbot with Batch Messages")
#     print("=" * 40)
    
#     for message in test_messages:
#         print(f"\n👤 Testing: '{message}'")
        
#         try:
#             payload = {"message": message}
#             response = requests.post(
#                 f"{BASE_URL}/chat",
#                 json=payload,
#                 headers={"Content-Type": "application/json"}
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 print(f"🤖 Response: {result['response']}")
#             else:
#                 print(f"❌ Error: {response.status_code}")
                
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Connection error: {e}")
    
#     print("\n" + "=" * 40)

# def test_health_endpoint():
#     """Test the health check endpoint"""
#     print("🏥 Testing Health Endpoint")
#     print("=" * 40)
    
#     try:
#         response = requests.get(f"{BASE_URL}/health")
#         if response.status_code == 200:
#             print("✅ Health check passed!")
#             print(f"   Response: {response.json()}")
#         else:
#             print(f"❌ Health check failed: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Connection error: {e}")
    
#     print("=" * 40)

# def test_root_endpoint():
#     """Test the root endpoint"""
#     print("🏠 Testing Root Endpoint")
#     print("=" * 40)
    
#     try:
#         response = requests.get(f"{BASE_URL}/")
#         if response.status_code == 200:
#             print("✅ Root endpoint working!")
#             print(f"   Response: {response.json()}")
#         else:
#             print(f"❌ Root endpoint failed: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Connection error: {e}")
    
#     print("=" * 40)

# if __name__ == "__main__":
#     print("🚀 Starting ZUS Coffee Chatbot Tests")
#     print("Make sure your FastAPI server is running on localhost:8000")
#     print("Run: python main.py")
#     print("\n")
    
#     # Test basic endpoints first
#     test_root_endpoint()
#     test_health_endpoint()
    
#     # Choose test mode
#     print("\nChoose test mode:")
#     print("1. Interactive chat")
#     print("2. Batch test")
    
#     choice = input("Enter 1 or 2: ").strip()
    
#     if choice == "1":
#         test_chatbot_interactive()
#     elif choice == "2":
#         test_chatbot_batch()
#     else:
#         print("Invalid choice. Running batch test...")
#         test_chatbot_batch()
    
#     print("\n🎉 Tests completed!")


# main.py - FastAPI app with streaming support
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
from llm import llm_groq

llm = llm_groq()

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
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
    message = data.get("message", "")
    
    async def generate_response():
        try:
            # Check if your LLM supports streaming
            if hasattr(llm, 'stream') or hasattr(llm, 'astream'):
                # If using async streaming
                if hasattr(llm, 'astream'):
                    async for chunk in llm.astream(message):
                        # Format as Server-Sent Events
                        chunk_data = {
                            "type": "chunk",
                            "content": str(chunk),
                            "user_message": message
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                # If using sync streaming
                elif hasattr(llm, 'stream'):
                    for chunk in llm.stream(message):
                        chunk_data = {
                            "type": "chunk", 
                            "content": str(chunk),
                            "user_message": message
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        # Add small delay for better UX
                        await asyncio.sleep(0.01)
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
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
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