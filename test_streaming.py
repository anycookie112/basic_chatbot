# test_streaming.py - Python client to test streaming
import requests
import json
import sys

def test_streaming_chat(message):
    url = "http://localhost:8000/chat/stream"
    data = {"message": message}
    
    print(f"Sending message: {message}")
    print("Response:")
    print("-" * 50)
    
    try:
        response = requests.post(
            url,
            json=data,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        json_data = line[6:]  # Remove "data: " prefix
                        try:
                            chunk = json.loads(json_data)
                            if chunk["type"] == "chunk":
                                print(chunk["content"], end="", flush=True)
                            elif chunk["type"] == "complete":
                                print(f"\n\n[Stream completed: {chunk['message']}]")
                                break
                            elif chunk["type"] == "error":
                                print(f"\n[Error: {chunk['message']}]")
                                break
                        except json.JSONDecodeError:
                            print(f"[Invalid JSON: {json_data}]")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def interactive_chat():
    print("ZUS Coffee Chatbot - Streaming Test")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            message = input("\nYou: ").strip()
            if message.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if message:
                print("Bot: ", end="", flush=True)
                test_streaming_chat(message)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single message mode
        message = " ".join(sys.argv[1:])
        test_streaming_chat(message)
    else:
        # Interactive mode
        interactive_chat()



"""

so client send a request to /chat/stream
fastapi will call the chat_stream_endpoint
generator function will stream the output.
the llm generates the output in chunks and formated into json and yielded (if return is used, will return the answer in 1 go, while yield used with async will stream the data)
fastapi will send the chunks in realtime









"""