# # test_streaming.py - Python client to test streaming
# import requests
# import json
# import sys




# def test_streaming_chat(message):
#     url = "http://localhost:8000/chat/stream"
#     prompt =  {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": message,
#                 }
#             ]
#         }
#     print(f"Sending message: {message}")
#     print("Response:")
#     print("-" * 50)
    
#     try:
#         response = requests.post(
#             url,
#             json=prompt,
#             stream=True,
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             for line in response.iter_lines(decode_unicode=True):
#                 if line:
#                     if line.startswith("data: "):
#                         json_data = line[6:]  # Remove "data: " prefix
#                         try:
#                             chunk = json.loads(json_data)
#                             if chunk["type"] == "chunk":
#                                 print(chunk["content"], end="", flush=True)
#                             elif chunk["type"] == "complete":
#                                 print(f"\n\n[Stream completed: {chunk['message']}]")
#                                 break
#                             elif chunk["type"] == "error":
#                                 print(f"\n[Error: {chunk['message']}]")
#                                 break
#                         except json.JSONDecodeError:
#                             print(f"[Invalid JSON: {json_data}]")
#         else:
#             print(f"Error: {response.status_code}")
#             print(response.text)
            
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")

# def interactive_chat():
#     print("ZUS Coffee Chatbot - Streaming Test")
#     print("Type 'quit' to exit")
#     print("=" * 50)
    
#     while True:
#         try:
#             message = input("\nYou: ").strip()
#             if message.lower() in ['quit', 'exit', 'q']:
#                 print("Goodbye!")
#                 break
            
#             if message:
#                 print("Bot: ", end="", flush=True)
#                 test_streaming_chat(message)
                
#         except KeyboardInterrupt:
#             print("\nGoodbye!")
#             break

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         # Single message mode
#         message = " ".join(sys.argv[1:])
#         test_streaming_chat(message)
#     else:
#         # Interactive mode
#         interactive_chat()



# """

# so client send a request to /chat/stream
# fastapi will call the chat_stream_endpoint
# generator function will stream the output.
# the llm generates the output in chunks and formated into json and yielded (if return is used, will return the answer in 1 go, while yield used with async will stream the data)
# fastapi will send the chunks in realtime




# """


# import sys
# import requests



# API_URL = "http://localhost:8000/chat"

# def test_chat(message):
#     prompt =  {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": message,
#                 }
#             ]
#         }
    
#     response = requests.post(API_URL, json= prompt)
#     if response.status_code == 200:
#         print("Bot:", response.json())
#     else:
#         print("Error:", response.status_code, response.text)

# def interactive_chat():
#     print("Interactive Chat with FastAPI Bot (type 'exit' to quit)")
#     while True:
#         msg = input("You: ")
#         if msg.lower() in ["exit", "quit"]:
#             break
#         test_chat(msg)

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         # Single message mode
#         message = " ".join(sys.argv[1:])
#         test_chat(message)
#     else:
#         # Interactive mode
#         interactive_chat()

# import sys
# import requests
# import json
# import time
# from typing import Generator

# API_URL = "http://localhost:8000/chat"
# STREAMING_API_URL = "http://localhost:8000/chat"  # Updated streaming endpoint
# SSE_API_URL = "http://localhost:8000/chat-sse"
# WEBSOCKET_URL = "ws://localhost:8000/ws/chat"

# def test_chat_non_streaming(message):
#     """Test non-streaming chat endpoint"""
#     prompt = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": message,
#             }
#         ]
#     }
    
#     response = requests.post(API_URL, json=prompt)
#     if response.status_code == 200:
#         print("Bot:", response.json())
#     else:
#         print("Error:", response.status_code, response.text)

# def test_chat_streaming(message):
#     """Test streaming chat endpoint"""
#     prompt = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": message,
#             }
#         ]
#     }
    
#     print("Bot (streaming): ", end="", flush=True)
    
#     try:
#         with requests.post(STREAMING_API_URL, json=prompt, stream=True) as response:
#             if response.status_code == 200:
#                 for line in response.iter_lines():
#                     if line:
#                         line_text = line.decode('utf-8')
#                         if line_text.startswith('data: '):
#                             try:
#                                 data = json.loads(line_text[6:])  # Remove 'data: ' prefix
                                
#                                 if data['type'] == 'content':
#                                     print(data['content'], end="", flush=True)
#                                 elif data['type'] == 'message':
#                                     print(f"\n[{data['message_type']}] {data['content']}", end="", flush=True)
#                                 elif data['type'] == 'tool_call':
#                                     print(f"\n[Tool Call: {data['name']}]", end="", flush=True)
#                                 elif data['type'] == 'end':
#                                     print("\n[Stream Complete]")
#                                     break
#                                 elif data['type'] == 'error':
#                                     print(f"\n[Error: {data['error']}]")
#                                     break
#                             except json.JSONDecodeError:
#                                 print(f"\n[Invalid JSON: {line_text}]")
#                 print()  # New line at the end
#             else:
#                 print(f"Error: {response.status_code} {response.text}")
#     except requests.exceptions.RequestException as e:
#         print(f"Request error: {e}")

# def test_chat_sse(message):
#     """Test Server-Sent Events endpoint"""
#     prompt = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": message,
#             }
#         ]
#     }
    
#     print("Bot (SSE): ", end="", flush=True)
    
#     try:
#         with requests.post(SSE_API_URL, json=prompt, stream=True) as response:
#             if response.status_code == 200:
#                 for line in response.iter_lines():
#                     if line:
#                         line_text = line.decode('utf-8')
#                         if line_text.startswith('data: '):
#                             try:
#                                 data = json.loads(line_text[6:])
#                                 if 'content' in data:
#                                     print(data['content'], end="", flush=True)
#                             except json.JSONDecodeError:
#                                 continue
#                         elif line_text.startswith('event: end'):
#                             print("\n[Stream Complete]")
#                             break
#                         elif line_text.startswith('event: error'):
#                             print(f"\n[Error occurred]")
#                             break
#                 print()
#             else:
#                 print(f"Error: {response.status_code} {response.text}")
#     except requests.exceptions.RequestException as e:
#         print(f"Request error: {e}")

# def test_websocket_chat(message):
#     """Test WebSocket chat endpoint"""
#     try:
#         import websockets
#         import asyncio
        
#         async def websocket_test():
#             uri = WEBSOCKET_URL
#             async with websockets.connect(uri) as websocket:
#                 # Send message
#                 prompt = {
#                     "messages": [
#                         {
#                             "role": "user",
#                             "content": message,
#                         }
#                     ]
#                 }
#                 await websocket.send(json.dumps(prompt))
                
#                 print("Bot (WebSocket): ", end="", flush=True)
                
#                 # Receive streaming response
#                 while True:
#                     try:
#                         response = await websocket.recv()
#                         data = json.loads(response)
                        
#                         if data['type'] == 'content':
#                             print(data['content'], end="", flush=True)
#                         elif data['type'] == 'response':
#                             # Handle your supervisor's response structure
#                             if 'messages' in data['data']:
#                                 for msg in data['data']['messages']:
#                                     if msg.get('content'):
#                                         print(f"\n[{msg.get('type', 'unknown')}] {msg['content']}", end="", flush=True)
#                         elif data['type'] == 'complete':
#                             print("\n[Stream Complete]")
#                             break
#                         elif data['type'] == 'error':
#                             print(f"\n[Error: {data['error']}]")
#                             break
#                     except websockets.exceptions.ConnectionClosed:
#                         print("\n[Connection closed]")
#                         break
#                 print()
        
#         asyncio.run(websocket_test())
        
#     except ImportError:
#         print("WebSocket testing requires 'websockets' package. Install with: pip install websockets")
#     except Exception as e:
#         print(f"WebSocket error: {e}")

# def interactive_chat():
#     """Interactive chat with multiple streaming options"""
#     print("Interactive Chat with FastAPI Bot (type 'exit' to quit)")
#     print("Commands:")
#     print("  - Regular message: normal chat")
#     print("  - /stream <message>: test streaming endpoint")
#     print("  - /sse <message>: test Server-Sent Events")
#     print("  - /ws <message>: test WebSocket")
#     print("  - /non-stream <message>: test non-streaming endpoint")
#     print()
    
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
        
#         if user_input.startswith('/stream '):
#             message = user_input[8:]  # Remove '/stream ' prefix
#             test_chat_streaming(message)
#         elif user_input.startswith('/sse '):
#             message = user_input[5:]  # Remove '/sse ' prefix
#             test_chat_sse(message)
#         elif user_input.startswith('/ws '):
#             message = user_input[4:]  # Remove '/ws ' prefix
#             test_websocket_chat(message)
#         elif user_input.startswith('/non-stream '):
#             message = user_input[12:]  # Remove '/non-stream ' prefix
#             test_chat_non_streaming(message)
#         else:
#             # Default to streaming
#             test_chat_streaming(user_input)

# def benchmark_streaming_vs_non_streaming(message, iterations=3):
#     """Compare streaming vs non-streaming performance"""
#     print(f"Benchmarking with message: '{message}'")
#     print(f"Running {iterations} iterations each...\n")
    
#     # Test non-streaming
#     print("=== Non-Streaming ===")
#     start_time = time.time()
#     for i in range(iterations):
#         print(f"Iteration {i+1}:")
#         test_chat_non_streaming(message)
#     non_streaming_time = time.time() - start_time
#     print(f"Non-streaming total time: {non_streaming_time:.2f}s\n")
    
#     # Test streaming
#     print("=== Streaming ===")
#     start_time = time.time()
#     for i in range(iterations):
#         print(f"Iteration {i+1}:")
#         test_chat_streaming(message)
#     streaming_time = time.time() - start_time
#     print(f"Streaming total time: {streaming_time:.2f}s\n")
    
#     print(f"Performance comparison:")
#     print(f"Non-streaming average: {non_streaming_time/iterations:.2f}s per request")
#     print(f"Streaming average: {streaming_time/iterations:.2f}s per request")

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         command = sys.argv[1]
        
#         if command == "benchmark" and len(sys.argv) > 2:
#             message = " ".join(sys.argv[2:])
#             benchmark_streaming_vs_non_streaming(message)
#         elif command == "stream" and len(sys.argv) > 2:
#             message = " ".join(sys.argv[2:])
#             test_chat_streaming(message)
#         elif command == "sse" and len(sys.argv) > 2:
#             message = " ".join(sys.argv[2:])
#             test_chat_sse(message)
#         elif command == "ws" and len(sys.argv) > 2:
#             message = " ".join(sys.argv[2:])
#             test_websocket_chat(message)
#         elif command == "non-stream" and len(sys.argv) > 2:
#             message = " ".join(sys.argv[2:])
#             test_chat_non_streaming(message)
#         else:
#             # Single message mode (default to streaming)
#             message = " ".join(sys.argv[1:])
#             test_chat_streaming(message)
#     else:
#         # Interactive mode
#         interactive_chat()



import sys
import requests
import json

API_URL = "http://localhost:8000/chat"
INCREMENTAL_URL = "http://localhost:8000/chat-incremental"
DEBUG_URL = "http://localhost:8000/debug"

def test_streaming(message, endpoint_url=API_URL):
    """Test streaming with proper message handling"""
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ]
    }
    
    print(f"Streaming response for: '{message}'")
    print("=" * 50)
    
    try:
        with requests.post(endpoint_url, json=prompt, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            try:
                                data = json.loads(line_text[6:])
                                
                                if data['type'] == 'message':
                                    # Display the message with context
                                    name = data.get('name', 'unknown')
                                    message_type = data.get('message_type', 'unknown')
                                    content = data.get('content', '')
                                    
                                    print(f"\n[{name} ({message_type})]:")
                                    print(content)
                                    print("-" * 30)
                                
                                elif data['type'] == 'content':
                                    # Display incremental content
                                    from_agent = data.get('from', 'unknown')
                                    content = data.get('content', '')
                                    
                                    print(f"\n[{from_agent}]: {content}")
                                
                                elif data['type'] == 'complete':
                                    print("\n✅ [Stream Complete]")
                                    break
                                    
                                elif data['type'] == 'error':
                                    print(f"\n❌ [Error: {data['error']}]")
                                    break
                                    
                            except json.JSONDecodeError as e:
                                print(f"\n⚠️  [JSON Error: {e}]")
                                
            else:
                print(f"❌ HTTP Error: {response.status_code} {response.text}")
                
    except Exception as e:
        print(f"❌ Request error: {e}")

def test_debug(message):
    """Test debug endpoint"""
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ]
    }
    
    print(f"Debug info for: '{message}'")
    print("=" * 50)
    
    response = requests.post(DEBUG_URL, json=prompt)
    if response.status_code == 200:
        debug_data = response.json()
        print(f"Total chunks: {debug_data['total_chunks']}")
        print(f"Message counts per chunk: {debug_data['message_counts']}")
        
        print("\nSample chunks:")
        for i, chunk in enumerate(debug_data['sample_chunks']):
            print(f"\nChunk {i+1}:")
            if 'supervisor' in chunk and 'messages' in chunk['supervisor']:
                messages = chunk['supervisor']['messages']
                print(f"  Messages in chunk: {len(messages)}")
                for j, msg in enumerate(messages[-2:]):  # Show last 2 messages
                    content = msg.get('content', '')[:100]  # First 100 chars
                    print(f"    Message {j+1}: {msg.get('name', 'unknown')} - {content}...")
    else:
        print(f"❌ Debug error: {response.status_code} {response.text}")

def interactive_chat():
    """Interactive chat with multiple options"""
    print("🤖 Supervisor Streaming Chat")
    print("Commands:")
    print("  /debug <message>     - Show debug info")
    print("  /incremental <message> - Stream incremental content")
    print("  /regular <message>   - Stream full messages")
    print("  <message>           - Default streaming")
    print("  exit                - Quit")
    print()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        if user_input.startswith('/debug '):
            message = user_input[7:]
            test_debug(message)
        elif user_input.startswith('/incremental '):
            message = user_input[13:]
            test_streaming(message, INCREMENTAL_URL)
        elif user_input.startswith('/regular '):
            message = user_input[9:]
            test_streaming(message, API_URL)
        else:
            test_streaming(user_input, API_URL)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "debug" and len(sys.argv) > 2:
            message = " ".join(sys.argv[2:])
            test_debug(message)
        elif command == "incremental" and len(sys.argv) > 2:
            message = " ".join(sys.argv[2:])
            test_streaming(message, INCREMENTAL_URL)
        else:
            message = " ".join(sys.argv[1:])
            test_streaming(message, API_URL)
    else:
        interactive_chat()