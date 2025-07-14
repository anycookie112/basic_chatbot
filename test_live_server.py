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
from customer_service_agent import cs_agent

from langchain_core.messages import HumanMessage, SystemMessage
def interactive_test():
    cs_agent_workflow = cs_agent()
    
    print("🤖 ZUS Coffee Chatbot - Interactive Test")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    thread_id = "main_conversation"  # Same thread for conversation continuity
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Add a system message to reset tool selection context
            reset_message = SystemMessage(content="""
            Analyze this new user question independently. 
            Choose the appropriate tool based ONLY on this current question:
            - call_product: for product/coffee/mug/food questions
            - call_outlet: for store/location/address/hours questions
            Ignore previous tool usage patterns.
            """)
            
            messages = [reset_message, HumanMessage(content=user_input)]
            print(config)
            result = cs_agent_workflow.invoke({"messages": messages}, config)
            last_message = result["messages"][-1]
            
            print(f"🤖 Bot: {last_message.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    interactive_test()