import requests
import json
import sys

# Server Configuration
SERVER_IP = "49.235.153.73"
PORT = 8000
BASE_URL = f"http://{SERVER_IP}:{PORT}"

def test_health():
    """Test the health check endpoint"""
    print(f"🔍 Testing connection to {BASE_URL}...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Health Check Passed: {response.json()}")
            return True
        else:
            print(f"❌ Health Check Failed: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Failed: Could not connect to {SERVER_IP}:{PORT}")
        print("   Please check if the firewall allows traffic on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_trade_execution():
    """Test the trade execution endpoint"""
    print(f"\n🚀 Testing Trade Execution...")
    
    trade_payload = {
        "action": "BUY",
        "symbol": "EURUSD",
        "volume": 0.01,
        "sl": 1.0500,
        "tp": 1.0700
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/trade/execute", 
            json=trade_payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ Trade Command Sent Successfully")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Trade Command Failed: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Trade Request Error: {e}")

if __name__ == "__main__":
    print("="*50)
    print("AlphaOS Bridge Remote Connection Test")
    print("="*50)
    
    if test_health():
        test_trade_execution()
    else:
        print("\n⚠️  Aborting trade test due to health check failure.")
        sys.exit(1)

