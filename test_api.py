#!/usr/bin/env python3
"""Quick API test script"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Model info: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing API endpoints...")
    
    if test_health():
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")
    
    if test_model_info():
        print("✅ Model info passed")
    else:
        print("❌ Model info failed")