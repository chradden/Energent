#!/usr/bin/env python3
"""
Test script for smartENERGY API integration in Agent B
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_b_price_forecast import AgentBPriceForecast
import pandas as pd
from datetime import datetime

def test_smartenergy_integration():
    """Test the smartENERGY API integration"""
    print("🧪 Testing smartENERGY API Integration")
    print("=" * 50)
    
    # Initialize Agent B
    agent_b = AgentBPriceForecast(model_type="transformer")
    
    # Test 1: Fetch data from smartENERGY
    print("\n1️⃣ Testing smartENERGY data fetching...")
    try:
        smartenergy_data = agent_b.get_smartenergy_data()
        print(f"✅ Successfully fetched {len(smartenergy_data)} data points")
        print(f"📊 Data range: {smartenergy_data['timestamp'].min()} to {smartenergy_data['timestamp'].max()}")
        print(f"💰 Price range: €{smartenergy_data['price'].min():.2f} - €{smartenergy_data['price'].max():.2f}")
        print(f"📈 Average price: €{smartenergy_data['price'].mean():.2f}")
        
        # Show first few rows
        print("\n📋 Sample data:")
        print(smartenergy_data.head())
        
    except Exception as e:
        print(f"❌ Error fetching smartENERGY data: {e}")
        return False
    
    # Test 2: Test current prices method
    print("\n2️⃣ Testing get_current_prices method...")
    try:
        current_prices = agent_b.get_current_prices()
        print("✅ Current prices method executed successfully")
        
        # Check which data sources are available
        available_sources = [source for source, data in current_prices.items() if data is not None]
        print(f"📡 Available data sources: {available_sources}")
        
        if 'smartenergy' in available_sources:
            print("✅ smartENERGY is available as primary source")
        else:
            print("⚠️ smartENERGY not available, using fallback sources")
            
    except Exception as e:
        print(f"❌ Error in get_current_prices: {e}")
        return False
    
    # Test 3: Test model training with smartENERGY data
    print("\n3️⃣ Testing model training with smartENERGY data...")
    try:
        agent_b.train(smartenergy_data)
        print("✅ Model trained successfully with smartENERGY data")
        
        # Test prediction
        forecast = agent_b.predict(smartenergy_data)
        print(f"🔮 Generated 24-hour forecast with {len(forecast)} data points")
        print(f"📊 Forecast range: €{forecast.min():.2f} - €{forecast.max():.2f}")
        
    except Exception as e:
        print(f"❌ Error in model training/prediction: {e}")
        return False
    
    # Test 4: Test model evaluation
    print("\n4️⃣ Testing model evaluation...")
    try:
        metrics = agent_b.evaluate(smartenergy_data)
        print("✅ Model evaluation completed")
        print(f"📈 Performance metrics:")
        print(f"   - MAE: {metrics['mae']:.2f} €/MWh")
        print(f"   - RMSE: {metrics['rmse']:.2f} €/MWh")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {e}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("✅ smartENERGY API integration is working correctly")
    return True

def test_api_endpoint():
    """Test the actual API endpoint"""
    print("\n🌐 Testing smartENERGY API endpoint directly...")
    
    import requests
    from datetime import datetime, timedelta
    
    url = "https://apis.smartenergy.at/market/v1/price"
    
    headers = {
        'User-Agent': 'ENERGENT-Test/1.0',
        'Accept': 'application/json'
    }
    
    params = {
        'region': 'AT',
        'start_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        'end_date': datetime.now().strftime('%Y-%m-%d'),
        'resolution': 'hourly'
    }
    
    try:
        print(f"🔗 Testing URL: {url}")
        print(f"📋 Parameters: {params}")
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        print(f"📡 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API response received successfully")
            print(f"📊 Response type: {type(data)}")
            print(f"📈 Response size: {len(str(data))} characters")
            
            # Show a sample of the response
            if isinstance(data, list) and len(data) > 0:
                print(f"📋 Sample response item: {data[0]}")
            elif isinstance(data, dict):
                print(f"📋 Response keys: {list(data.keys())}")
                
        else:
            print(f"❌ API returned error status: {response.status_code}")
            print(f"📄 Response text: {response.text[:200]}...")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting smartENERGY API Integration Tests")
    print("=" * 60)
    
    # Test the API endpoint first
    test_api_endpoint()
    
    # Test the integration
    success = test_smartenergy_integration()
    
    if success:
        print("\n🎯 Integration test completed successfully!")
        print("✅ Agent B is now configured to use smartENERGY as primary data source")
    else:
        print("\n⚠️ Integration test encountered issues")
        print("🔧 Please check the API endpoint and network connectivity") 