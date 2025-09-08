#!/usr/bin/env python3
"""
Test script to verify FastVLM model loading and processing.
This script helps verify if the model is actually loaded and processing images,
or if it's falling back to mock mode.
"""

import requests
import json
import time
from datetime import datetime
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import sys

API_BASE = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_api_status():
    """Check if API is running and get model status"""
    print_section("API Status Check")
    try:
        response = requests.get(f"{API_BASE}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            
            # Print model status
            model_info = data.get('model', {})
            print(f"\n📊 Model Information:")
            print(f"  - Loaded: {model_info.get('is_loaded', False)}")
            print(f"  - Type: {model_info.get('model_type', 'unknown')}")
            print(f"  - Model Name: {model_info.get('model_name', 'N/A')}")
            print(f"  - Device: {model_info.get('device', 'unknown')}")
            print(f"  - Parameters: {model_info.get('parameters_count', 0) / 1e9:.2f}B")
            
            if model_info.get('error'):
                print(f"  - Error: {model_info['error']}")
                
            if model_info.get('loading_time'):
                print(f"  - Loading Time: {model_info['loading_time']:.2f}s")
                
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def get_model_status():
    """Get detailed model status"""
    print_section("Detailed Model Status")
    try:
        response = requests.get(f"{API_BASE}/model/status")
        if response.status_code == 200:
            status = response.json()
            print(json.dumps(status, indent=2))
            return status
        else:
            print(f"❌ Failed to get model status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting model status: {e}")
        return None

def test_model_endpoint():
    """Test the model with a synthetic image"""
    print_section("Testing Model with Synthetic Image")
    try:
        response = requests.post(f"{API_BASE}/model/test")
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Test completed successfully")
            print(f"\n📷 Test Image: {result['test_image_size']}")
            
            analysis = result['analysis_result']
            print(f"\n🔍 Analysis Results:")
            print(f"  Summary: {analysis['summary'][:200]}...")
            
            if analysis.get('mock_mode'):
                print(f"  ⚠️  WARNING: Model is running in MOCK MODE")
                print(f"     No actual vision-language model is loaded!")
            else:
                print(f"  ✅ Real model is processing images")
                
            print(f"\n  UI Elements Detected: {len(analysis.get('ui_elements', []))}")
            for elem in analysis.get('ui_elements', [])[:3]:
                print(f"    - {elem.get('type')}: {elem.get('text')}")
                
            print(f"\n  Text Snippets: {len(analysis.get('text_snippets', []))}")
            for text in analysis.get('text_snippets', [])[:3]:
                print(f"    - {text}")
                
            if analysis.get('model_info'):
                model_info = analysis['model_info']
                print(f"\n  Model Used: {model_info.get('model_type')} - {model_info.get('model_name', 'N/A')}")
                
            return result
        else:
            print(f"❌ Test failed with status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return None

def test_real_screenshot():
    """Test with a real screenshot"""
    print_section("Testing with Real Screenshot")
    
    # Create a more complex test image
    img = Image.new('RGB', (1920, 1080), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Draw a mock browser window
    draw.rectangle([0, 0, 1920, 80], fill='#333333')  # Title bar
    draw.text((50, 30), "FastVLM Screen Observer - Test Page", fill='white')
    
    # Draw some UI elements
    draw.rectangle([100, 150, 400, 200], fill='#4CAF50', outline='#45a049')
    draw.text((200, 165), "Click Me", fill='white')
    
    draw.rectangle([100, 250, 600, 300], fill='white', outline='#ddd')
    draw.text((110, 265), "Enter your email address...", fill='#999')
    
    draw.rectangle([100, 350, 250, 400], fill='#2196F3', outline='#1976D2')
    draw.text((140, 365), "Submit", fill='white')
    
    # Add some text content
    draw.text((100, 450), "Welcome to FastVLM Screen Observer", fill='#333')
    draw.text((100, 480), "This is a test page to verify model functionality", fill='#666')
    draw.text((100, 510), "The model should detect buttons, text fields, and content", fill='#666')
    
    # Add a warning box
    draw.rectangle([700, 150, 1200, 250], fill='#FFF3CD', outline='#FFC107')
    draw.text((720, 170), "⚠️ Warning: This is sensitive information", fill='#856404')
    draw.text((720, 200), "Credit Card: **** **** **** 1234", fill='#856404')
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Send to API
    try:
        payload = {
            "image_data": f"data:image/png;base64,{img_str}",
            "include_thumbnail": False
        }
        
        response = requests.post(f"{API_BASE}/analyze", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed")
            print(f"\n📝 Summary: {result['summary']}")
            
            if "[MOCK MODE]" in result['summary']:
                print(f"\n⚠️  WARNING: Analysis is using MOCK MODE")
                print(f"   Install a real vision-language model for actual analysis")
            else:
                print(f"\n✅ Real model analysis completed")
                
            print(f"\n🔍 Detected Elements:")
            print(f"  - UI Elements: {len(result.get('ui_elements', []))}")
            print(f"  - Text Snippets: {len(result.get('text_snippets', []))}")
            print(f"  - Risk Flags: {result.get('risk_flags', [])}")
            
            return result
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error analyzing screenshot: {e}")
        return None

def try_reload_model(model_type="blip"):
    """Try to reload the model with a specific type"""
    print_section(f"Attempting to Load {model_type.upper()} Model")
    
    try:
        print(f"🔄 Requesting model reload with type: {model_type}")
        response = requests.post(f"{API_BASE}/model/reload?model_type={model_type}")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ Model loaded successfully!")
                status = result['status']
                print(f"  - Model: {status['model_name']}")
                print(f"  - Device: {status['device']}")
                print(f"  - Loading Time: {status.get('loading_time', 0):.2f}s")
            else:
                print(f"❌ Failed to load model")
                print(f"  - Error: {result['status'].get('error')}")
            return result
        else:
            print(f"❌ Reload request failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error reloading model: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("  FastVLM Model Verification Test")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    # Step 1: Check API status
    if not check_api_status():
        print("\n❌ API is not running. Please start the backend first.")
        print("   Run: cd backend && ./start.sh")
        return
    
    # Step 2: Get detailed model status
    model_status = get_model_status()
    
    # Step 3: Test with synthetic image
    test_result = test_model_endpoint()
    
    # Step 4: Test with complex screenshot
    screenshot_result = test_real_screenshot()
    
    # Step 5: If in mock mode, try loading a lightweight model
    if model_status and model_status.get('model_type') == 'mock':
        print_section("Model Loading Recommendations")
        print("\n⚠️  The system is currently running in MOCK MODE")
        print("   No actual vision-language model is loaded.\n")
        print("   To load a real model, you can:")
        print("   1. Install required dependencies:")
        print("      pip install transformers torch torchvision")
        print("   2. Try loading BLIP (lightweight, ~400MB):")
        print("      curl -X POST http://localhost:8000/model/reload?model_type=blip")
        print("   3. Or try LLaVA (more capable, ~7GB):")
        print("      curl -X POST http://localhost:8000/model/reload?model_type=llava")
        
        # Offer to try loading BLIP
        print("\n🤖 Would you like to try loading BLIP model now?")
        print("   (This will download ~400MB and may take a minute)")
        try:
            response = input("   Load BLIP? (y/n): ").strip().lower()
            if response == 'y':
                try_reload_model("blip")
                # Re-test after loading
                print("\n🔄 Re-testing with new model...")
                test_model_endpoint()
        except KeyboardInterrupt:
            print("\n   Skipped model loading")
    
    print_section("Test Complete")
    
    if model_status and model_status.get('is_loaded') and model_status.get('model_type') != 'mock':
        print("\n✅ SUCCESS: Real vision-language model is loaded and processing images!")
        print(f"   Model: {model_status.get('model_name')}")
        print(f"   Type: {model_status.get('model_type')}")
    else:
        print("\n⚠️  System is running in MOCK MODE")
        print("   Install and load a real model for actual image analysis")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()