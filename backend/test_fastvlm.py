#!/usr/bin/env python3
"""
Test script for FastVLM-7B model loading and configuration
"""

import asyncio
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    deps = {
        "torch": None,
        "transformers": None,
        "sentencepiece": None,
        "einops": None,
        "accelerate": None
    }
    
    for dep in deps:
        try:
            module = __import__(dep)
            deps[dep] = getattr(module, "__version__", "installed")
            print(f"✓ {dep}: {deps[dep]}")
        except ImportError:
            print(f"✗ {dep}: NOT INSTALLED")
            deps[dep] = None
    
    return all(v is not None for v in deps.values())

def check_hardware():
    """Check hardware capabilities"""
    print("\nHardware check:")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        print("✓ Apple Silicon MPS available")
        # Get system memory
        import subprocess
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            mem_bytes = int(result.stdout.split()[1])
            print(f"  System Memory: {mem_bytes / 1e9:.2f} GB")
    else:
        print("✓ CPU mode")
        import psutil
        print(f"  Available Memory: {psutil.virtual_memory().available / 1e9:.2f} GB")

async def test_fastvlm_loading():
    """Test loading FastVLM-7B model"""
    print("\n" + "="*50)
    print("Testing FastVLM-7B Model Loading")
    print("="*50)
    
    model_name = "apple/FastVLM-7B"
    
    try:
        print(f"\n1. Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        print("   ✓ Tokenizer loaded successfully")
        print(f"   Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        
        # Check for IMAGE_TOKEN_INDEX
        IMAGE_TOKEN_INDEX = -200
        if hasattr(tokenizer, 'IMAGE_TOKEN_INDEX'):
            print(f"   IMAGE_TOKEN_INDEX: {tokenizer.IMAGE_TOKEN_INDEX}")
        else:
            print(f"   Setting IMAGE_TOKEN_INDEX to {IMAGE_TOKEN_INDEX}")
            tokenizer.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        
        print("\n2. Attempting to load model...")
        print("   Note: This requires ~14GB RAM for full precision")
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        
        print(f"   Device: {device}")
        print(f"   Dtype: {dtype}")
        
        # Try loading with minimal memory usage
        print("   Loading with low_cpu_mem_usage=True...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        print("   ✓ Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params / 1e9:.2f}B")
        
        # Move to device
        print(f"\n3. Moving model to {device}...")
        model = model.to(device)
        model.eval()
        print("   ✓ Model ready for inference")
        
        # Test a simple generation
        print("\n4. Testing generation...")
        test_prompt = "Hello, this is a test of"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Input: {test_prompt}")
        print(f"   Output: {response}")
        
        print("\n✓ FastVLM-7B is working correctly!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        if "trust_remote_code" in str(e):
            print("\nSolution: The model requires trust_remote_code=True")
            print("This is already set in the code, but the model files may need to be re-downloaded.")
        return False
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n✗ Out of Memory Error")
            print("\nSolutions:")
            print("1. Use the quantized version:")
            print("   model_name = 'apple/FastVLM-7B-int4'")
            print("2. Use a smaller variant:")
            print("   model_name = 'apple/FastVLM-1.5B'")
            print("3. Enable 8-bit quantization (requires bitsandbytes)")
            print("4. Increase system RAM or use a GPU")
        else:
            print(f"\n✗ Runtime Error: {e}")
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_alternative_models():
    """Test alternative model options if FastVLM-7B fails"""
    print("\n" + "="*50)
    print("Alternative Model Options")
    print("="*50)
    
    alternatives = [
        ("apple/FastVLM-1.5B", "Smaller FastVLM variant (1.5B params)"),
        ("apple/FastVLM-7B-int4", "Quantized FastVLM for lower memory"),
        ("apple/FastVLM-0.5B", "Smallest FastVLM variant (0.5B params)")
    ]
    
    for model_name, description in alternatives:
        print(f"\n• {model_name}")
        print(f"  {description}")
        try:
            # Just check if the model card exists
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print(f"  ✓ Model available")
        except Exception as e:
            print(f"  ✗ Not accessible: {str(e)[:50]}...")

async def main():
    """Main test function"""
    print("FastVLM-7B Integration Test")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install all requirements.")
        return
    
    # Check hardware
    check_hardware()
    
    # Test FastVLM loading
    success = await test_fastvlm_loading()
    
    if not success:
        # Show alternatives
        await test_alternative_models()
        
        print("\n" + "="*50)
        print("Recommendations:")
        print("="*50)
        print("\n1. If memory is limited, use FastVLM-1.5B or FastVLM-0.5B")
        print("2. For Apple Silicon, ensure you have enough RAM (16GB+ recommended)")
        print("3. Consider using the quantized version (FastVLM-7B-int4)")
        print("4. Make sure transformers >= 4.40.0 is installed")

if __name__ == "__main__":
    asyncio.run(main())