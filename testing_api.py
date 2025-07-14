#!/usr/bin/env python3
"""
Script to test API endpoints sequentially to debug the "first message works" issue
"""

import requests
import json
import time

def test_sequential_requests(url, messages, endpoint_name):
    """Test multiple sequential requests to the same endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing {endpoint_name} with sequential requests")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Request {i} ---")
        print(f"Message: {message}")
        
        payload = {"message": message}
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {end_time - start_time:.2f} seconds")
            print(f"Response Length: {len(response.text)} characters")
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    answer = json_response.get('answer', 'No answer field')
                    print(f"✅ Success: {answer[:100]}...")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"Raw response: {response.text}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Error response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection Error: Cannot connect to {url}")
            print("   API server might have crashed or stopped responding")
        except requests.exceptions.Timeout:
            print(f"❌ Timeout Error: Request took longer than 15 seconds")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        # Small delay between requests
        time.sleep(1)

def test_alternating_endpoints():
    """Test alternating between product and outlet endpoints"""
    print(f"\n{'='*60}")
    print("Testing alternating between endpoints")
    print(f"{'='*60}")
    
    PRODUCT_API = "http://localhost:8000/product"
    OUTLET_API = "http://localhost:8000/outlet"
    
    test_sequence = [
        (PRODUCT_API, "what mugs do you sell?", "PRODUCT"),
        (OUTLET_API, "show me 1 store in ampang", "OUTLET"),
        (PRODUCT_API, "what coffee do you have?", "PRODUCT"),
        (OUTLET_API, "stores in KL", "OUTLET"),
    ]
    
    for i, (url, message, endpoint_type) in enumerate(test_sequence, 1):
        print(f"\n--- Test {i}: {endpoint_type} ---")
        print(f"URL: {url}")
        print(f"Message: {message}")
        
        payload = {"message": message}
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {end_time - start_time:.2f} seconds")
            
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    answer = json_response.get('answer', 'No answer field')
                    print(f"✅ Success: {answer[:100]}...")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Error response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)  # Longer delay between different endpoints

def main():
    """Run all tests"""
    print("🧪 Testing API Sequential Request Handling")
    
    # Test same endpoint multiple times
    PRODUCT_API = "http://localhost:8000/product"
    OUTLET_API = "http://localhost:8000/outlet"
    
    product_messages = [
        "what mugs do you sell?",
        "show me coffee products",
        "what's the price of cups?",
        "do you have food items?"
    ]
    
    outlet_messages = [
        "show me 1 store in ampang",
        "stores in KL",
        "outlet addresses",
        "opening hours"
    ]
    
    # Test product endpoint
    # test_sequential_requests(PRODUCT_API, product_messages, "PRODUCT API")
    
    # print("\n" + "="*60)
    # print("Waiting 5 seconds before testing outlet API...")
    # print("="*60)
    # time.sleep(5)
    
    # Test outlet endpoint
    test_sequential_requests(OUTLET_API, outlet_messages, "OUTLET API")
    
    print("\n" + "="*60)
    print("Waiting 5 seconds before testing alternating endpoints...")
    print("="*60)
    time.sleep(5)
    
    # Test alternating endpoints
    test_alternating_endpoints()

if __name__ == "__main__":
    main()