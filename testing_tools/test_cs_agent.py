# METHOD 1: Direct Testing (No API needed)
# test_cs_agent.py
from customer_service_agent import cs_agent

from langchain_core.messages import HumanMessage

# Initialize your cs_agent
cs_agent_workflow = cs_agent()

def test_direct():
    """Test cs_agent directly without API"""
    test_messages = [
        "Hello!",
        "What products do you have?",
        "Are there any outlets in KL?",
        "Show me mugs available at KL outlets"
    ]
    
    for message in test_messages:
        print(f"\n{'='*50}")
        print(f"USER: {message}")
        print(f"{'='*50}")
        
        # Test the workflow
        messages = [HumanMessage(content=message)]
        result = cs_agent_workflow.invoke({"messages": messages})
        
        # Get the response
        last_message = result["messages"][-1]
        print(f"BOT: {last_message.content}")
        
        # Show which tools were called
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"TOOLS USED: {[tc['name'] for tc in last_message.tool_calls]}")

if __name__ == "__main__":
    test_direct()


# METHOD 2: API Testing with FastAPI
# test_api.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

app = FastAPI()
cs_agent_workflow = cs_agent()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        messages = [HumanMessage(content=request.message)]
        result = cs_agent_workflow.invoke({"messages": messages})
        last_message = result["messages"][-1]
        
        return {
            "response": last_message.content,
            "user_message": request.message
        }
    except Exception as e:
        return {"error": str(e)}

# Test the API
def test_api():
    client = TestClient(app)
    
    test_cases = [
        {"message": "Hello!"},
        {"message": "What coffee do you sell?"},
        {"message": "Where are your stores located?"},
        {"message": "Tell me about your mugs and store locations"}
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['message']}")
        print(f"{'='*50}")
        
        response = client.post("/chat", json=test_case)
        result = response.json()
        
        print(f"Response: {result}")

if __name__ == "__main__":
    test_api()


# METHOD 3: Live Server Testing
# Run this after starting your server
import requests
import json

def test_live_server():
    """Test against running server"""
    base_url = "http://localhost:8000"
    
    test_queries = [
        "Hi there!",
        "What drinks do you have?",
        "Show me your coffee menu",
        "Where is your nearest outlet?",
        "What are your opening hours?",
        "Do you have any mugs for sale?",
        "Tell me about outlets in Kuala Lumpur"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {query}")
        print(f"{'='*60}")
        
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"message": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f" SUCCESS")
                print(f" Response: {result['response']}")
            else:
                print(f" ERROR: {response.status_code}")
                print(f" Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f" CONNECTION ERROR: {e}")

if __name__ == "__main__":
    test_live_server()


# METHOD 4: Comprehensive Test Suite
# comprehensive_test.py
import unittest
from fastapi.testclient import TestClient

class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_general_conversation(self):
        """Test general conversation"""
        response = self.client.post("/chat", json={"message": "Hello!"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("response", result)
    
    def test_product_queries(self):
        """Test product-related queries"""
        product_queries = [
            "What coffee do you sell?",
            "Show me your menu",
            "Do you have mugs?",
            "What's your cheapest drink?"
        ]
        
        for query in product_queries:
            with self.subTest(query=query):
                response = self.client.post("/chat", json={"message": query})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertIn("response", result)
                self.assertIsInstance(result["response"], str)
    
    def test_outlet_queries(self):
        """Test outlet-related queries"""
        outlet_queries = [
            "Where are your stores?",
            "What are your opening hours?",
            "Do you have outlets in KL?",
            "Store locations near me"
        ]
        
        for query in outlet_queries:
            with self.subTest(query=query):
                response = self.client.post("/chat", json={"message": query})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertIn("response", result)
    
    def test_combined_queries(self):
        """Test queries that might use both tools"""
        combined_queries = [
            "What drinks are available at your KL outlets?",
            "Show me mugs and store locations",
            "Can I buy coffee at your nearest store?"
        ]
        
        for query in combined_queries:
            with self.subTest(query=query):
                response = self.client.post("/chat", json={"message": query})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertIn("response", result)
    
    def test_error_handling(self):
        """Test error handling"""
        response = self.client.post("/chat", json={"message": ""})
        # Should handle empty messages gracefully
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()


# METHOD 5: Interactive Testing
# interactive_test.py
from langchain_core.messages import HumanMessage

def interactive_test():
    """Interactive testing in terminal"""
    cs_agent_workflow = cs_agent()
    
    print("🤖 ZUS Coffee Chatbot - Interactive Test")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        try:
            messages = [HumanMessage(content=user_input)]
            result = cs_agent_workflow.invoke({"messages": messages})
            last_message = result["messages"][-1]
            
            print(f"🤖 Bot: {last_message.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_test()