#!/usr/bin/env python3
"""
Test script for PCOSense ML API
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Model: {data['model']}")
            print(f"  Accuracy: {data['accuracy']:.2%}")
            return True
        else:
            print(f"✗ Health check failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the Flask server running?")
        print("  Start it with: python flask_api.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting /predict endpoint...")
    
    # Test case 1: Moderate PCOS indicators
    test_data = {
        "age": 28,
        "bmi": 27,
        "cycle_length": 38,
        "cycle_regularity": 0,
        "hair_growth": 1,
        "pimples": 1,
        "hair_loss": 0,
        "weight_gain": 1,
        "skin_darkening": 1
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                pred = result['prediction']
                print(f"✓ Prediction successful")
                print(f"  Stage: {pred['stage']} - {pred['stage_name']}")
                print(f"  Confidence: {pred['confidence']:.2f}%")
                print(f"  Risk Factors: {len([k for k, v in result['risk_factors'].items() if v])} detected")
                print(f"  Yoga recommendations: {len(result['recommendations']['yoga'])}")
                print(f"  Nutrition tips: {len(result['recommendations']['nutrition'])}")
                print(f"  Exercise suggestions: {len(result['recommendations']['exercise'])}")
                return True
            else:
                print(f"✗ Prediction failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Prediction request failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_multiple_cases():
    """Test multiple scenarios"""
    print("\nTesting multiple scenarios...")
    
    test_cases = [
        {
            "name": "Healthy (No PCOS expected)",
            "data": {
                "age": 25,
                "bmi": 22,
                "cycle_length": 28,
                "cycle_regularity": 2,
                "hair_growth": 0,
                "pimples": 0,
                "hair_loss": 0,
                "weight_gain": 0,
                "skin_darkening": 0
            }
        },
        {
            "name": "Severe PCOS indicators",
            "data": {
                "age": 30,
                "bmi": 32,
                "cycle_length": 45,
                "cycle_regularity": 0,
                "hair_growth": 1,
                "pimples": 1,
                "hair_loss": 1,
                "weight_gain": 1,
                "skin_darkening": 1
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Case {i}: {test_case['name']}")
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test_case['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    pred = result['prediction']
                    print(f"    → {pred['stage_name']} ({pred['confidence']:.1f}% confidence)")
                else:
                    print(f"    → Failed: {result.get('error')}")
            else:
                print(f"    → HTTP {response.status_code}")
                
        except Exception as e:
            print(f"    → Error: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("PCOSense ML API Test Suite")
    print("=" * 60)
    print()
    
    # Test 1: Health check
    health_ok = test_health()
    
    if not health_ok:
        print("\n⚠️  API is not responding. Please start the Flask server first.")
        print("   Run: python flask_api.py")
        return
    
    # Test 2: Basic prediction
    test_prediction()
    
    # Test 3: Multiple cases
    test_multiple_cases()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()