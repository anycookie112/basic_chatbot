# import requests
# import json

# def test_live_server():
#     """Test against running server"""
#     base_url = "http://localhost:8001"
    
#     test_queries = [
#         "Hi there!",
#         "What drinks do you have?",
#         "Show me your coffee menu",
#         "Where is your nearest outlet?",
#         "What are your opening hours?",
#         "Do you have any mugs for sale?",
#         "Tell me about outlets in Kuala Lumpur"
#     ]
    
#     for query in test_queries:
#         print(f"\n{'='*60}")
#         print(f"🧪 TESTING: {query}")
#         print(f"{'='*60}")
        
#         try:
#             response = requests.post(
#                 f"{base_url}/chat",
#                 json={"message": query},
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 print(f"✅ SUCCESS")
#                 print(f"📝 Response: {result['response']}")
#             else:
#                 print(f"❌ ERROR: {response.status_code}")
#                 print(f"📝 Error: {response.text}")
                
#         except requests.exceptions.RequestException as e:
#             print(f"❌ CONNECTION ERROR: {e}")

# if __name__ == "__main__":
#     test_live_server()
from router_agent import cs_api

from langchain_core.messages import HumanMessage

# Initialize your cs_agent
cs_agent_workflow = cs_api()

from langchain_core.messages import HumanMessage

def interactive_test():
    """Interactive testing in terminal"""
    cs_agent_workflow = cs_api()
    
    print("🤖 ZUS Coffee Chatbot - Interactive Test")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        try:
            config = {"configurable": {"thread_id": "1"}}

        # Invoke your cs_agent workflow

            messages = [HumanMessage(content=user_input)]
            result = cs_agent_workflow.invoke({"messages": messages}, config)
            last_message = result["messages"][-1]
            
            print(f"🤖 Bot: {last_message.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_test()