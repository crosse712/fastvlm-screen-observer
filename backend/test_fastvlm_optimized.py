#!/usr/bin/env python3
"""
Test script for loading FastVLM with memory optimization
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fastvlm_model import FastVLMModel

async def test_fastvlm_auto():
    """Test automatic FastVLM model selection based on available memory"""
    print("="*50)
    print("Testing FastVLM with Automatic Model Selection")
    print("="*50)
    
    # Create model instance
    model = FastVLMModel()
    
    # Try loading with auto mode (will select based on available memory)
    print("\n1. Initializing model with auto selection...")
    await model.initialize(model_type="auto")
    
    # Check status
    status = model.get_status()
    print(f"\n2. Model Status:")
    print(f"   Loaded: {status['is_loaded']}")
    print(f"   Type: {status['model_type']}")
    print(f"   Name: {status['model_name']}")
    print(f"   Device: {status['device']}")
    print(f"   Parameters: {status['parameters_count'] / 1e9:.2f}B" if status['parameters_count'] > 0 else "   Parameters: N/A")
    
    if status['is_loaded'] and status['model_type'] != "mock":
        print("\n✓ FastVLM model loaded successfully!")
        print("   The system automatically selected the best model for your available memory.")
        
        # Test image analysis
        print("\n3. Testing image analysis...")
        from PIL import Image
        import io
        
        # Create a test image
        test_image = Image.new('RGB', (336, 336), color='red')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        result = await model.analyze_image(img_byte_arr)
        print(f"   Analysis result: {result.get('summary', 'No summary')[:100]}...")
        
    else:
        print(f"\n⚠ Model not fully loaded: {status.get('error', 'Unknown error')}")
    
    return status

async def test_specific_model(model_type: str):
    """Test loading a specific FastVLM variant"""
    print(f"\n{'='*50}")
    print(f"Testing {model_type} Model")
    print("="*50)
    
    # Create model instance
    model = FastVLMModel()
    
    # Try loading specific model
    print(f"\nLoading {model_type}...")
    await model.initialize(model_type=model_type)
    
    # Check status
    status = model.get_status()
    print(f"\nStatus: {'✓ Loaded' if status['is_loaded'] else '✗ Failed'}")
    if status['error']:
        print(f"Error: {status['error']}")
    
    return status

async def main():
    """Main test function"""
    print("FastVLM Integration Test - Optimized for Limited Memory")
    print("="*50)
    
    # Test automatic selection
    auto_status = await test_fastvlm_auto()
    
    # If auto didn't work, try specific smaller models
    if not auto_status['is_loaded'] or auto_status['model_type'] == "mock":
        print("\n" + "="*50)
        print("Trying Alternative Models")
        print("="*50)
        
        # Try smaller variants
        for model_type in ["fastvlm-small", "blip"]:
            status = await test_specific_model(model_type)
            if status['is_loaded']:
                print(f"\n✓ Successfully loaded {model_type} as fallback")
                break
    
    print("\n" + "="*50)
    print("Test Complete")
    print("="*50)
    
    if auto_status['is_loaded'] and auto_status['model_type'] != "mock":
        print("\n✓ SUCCESS: FastVLM is properly configured and working!")
        print(f"  Model: {auto_status['model_name']}")
        print(f"  Device: {auto_status['device']}")
        print("\nThe model is ready to use in your application.")
    else:
        print("\n⚠ WARNING: FastVLM could not be loaded with current memory.")
        print("\nRecommendations:")
        print("1. Free up system memory and try again")
        print("2. Use the BLIP model as a fallback (already working)")
        print("3. Consider upgrading to 16GB+ RAM for full FastVLM-7B")
        print("4. Use cloud GPU services for production deployment")

if __name__ == "__main__":
    asyncio.run(main())