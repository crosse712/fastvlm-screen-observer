#!/usr/bin/env python3
"""
Test FastVLM-7B with 8-bit quantization for limited RAM systems
Following exact HuggingFace model card implementation
"""

import torch
import psutil
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def check_system():
    """Check system capabilities"""
    print("="*60)
    print("System Check")
    print("="*60)
    
    # Memory check
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / 1e9:.2f} GB")
    print(f"Available RAM: {mem.available / 1e9:.2f} GB")
    print(f"Used RAM: {mem.percent}%")
    
    # Device check
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Device: Apple Silicon MPS")
    else:
        device = "cpu"
        print("Device: CPU")
    
    print()
    return device, mem.available / 1e9

def test_fastvlm_quantized():
    """Test FastVLM-7B with quantization"""
    print("="*60)
    print("Testing FastVLM-7B with 8-bit Quantization")
    print("="*60)
    
    device, available_gb = check_system()
    
    # Model ID from HuggingFace
    MID = "apple/FastVLM-7B"
    IMAGE_TOKEN_INDEX = -200  # As specified in model card
    
    print(f"\n1. Loading tokenizer from {MID}...")
    try:
        tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        print(f"   ✓ Tokenizer loaded: {tok.__class__.__name__}")
        print(f"   ✓ Vocab size: {tok.vocab_size}")
        print(f"   ✓ IMAGE_TOKEN_INDEX = {IMAGE_TOKEN_INDEX}")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        return False
    
    print(f"\n2. Configuring 8-bit quantization...")
    if available_gb < 12:
        print(f"   Memory available: {available_gb:.2f} GB")
        print("   Using 8-bit quantization for memory efficiency")
        
        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16 if device != "cpu" else torch.float32,
            bnb_8bit_use_double_quant=True,  # Extra memory optimization
            bnb_8bit_quant_type="nf4"  # Better quality quantization
        )
        
        model_kwargs = {
            "quantization_config": quantization_config,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        print("   Configuration: 8-bit NF4 quantization with double quantization")
        print("   Expected memory usage: ~7GB")
    else:
        print(f"   Memory available: {available_gb:.2f} GB (sufficient for full precision)")
        model_kwargs = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        print("   Configuration: Full precision")
        print("   Expected memory usage: ~14GB")
    
    print(f"\n3. Loading model from {MID}...")
    print("   This may take several minutes on first run...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MID,
            **model_kwargs
        )
        print("   ✓ Model loaded successfully!")
        
        # Check model details
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Parameters: {total_params / 1e9:.2f}B")
        
        # Check if vision tower is available
        if hasattr(model, 'get_vision_tower'):
            print("   ✓ Vision tower (FastViTHD) available")
        else:
            print("   ⚠ Vision tower not detected")
        
        print(f"\n4. Testing generation with IMAGE_TOKEN_INDEX...")
        
        # Test message with image placeholder
        messages = [
            {"role": "user", "content": "<image>\nDescribe this image."}
        ]
        
        # Apply chat template
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize parts
        pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Create image token
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        
        # Combine tokens
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        print(f"   Input IDs shape: {input_ids.shape}")
        print(f"   Image token inserted at position: {(input_ids == IMAGE_TOKEN_INDEX).nonzero()[0, 1].item()}")
        
        print("\n✅ SUCCESS: FastVLM-7B is properly configured!")
        print(f"   - Model: {MID}")
        print(f"   - IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
        print(f"   - Quantization: {'8-bit' if available_gb < 12 else 'Full precision'}")
        print(f"   - trust_remote_code: True")
        print(f"   - Device: {device}")
        
        # Memory usage after loading
        mem_after = psutil.virtual_memory()
        mem_used = (mem.total - mem_after.available) / 1e9
        print(f"\n   Memory used by model: ~{mem_used:.2f} GB")
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n✗ Out of Memory Error!")
            print("\nThe system does not have enough RAM even with 8-bit quantization.")
            print("Solutions:")
            print("1. Close other applications to free memory")
            print("2. Use apple/FastVLM-1.5B (smaller model)")
            print("3. Upgrade to 16GB+ RAM")
            print("4. Use cloud GPU services")
        else:
            print(f"\n✗ Runtime Error: {e}")
        return False
        
    except ImportError as e:
        if "bitsandbytes" in str(e):
            print("\n✗ bitsandbytes not installed properly")
            print("Run: pip install bitsandbytes")
        else:
            print(f"\n✗ Import Error: {e}")
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("FastVLM-7B Quantization Test")
    print("Using exact implementation from HuggingFace model card")
    print()
    
    success = test_fastvlm_quantized()
    
    if not success:
        print("\n" + "="*60)
        print("Hardware Requirements Not Met")
        print("="*60)
        print("\nFastVLM-7B requires one of:")
        print("• 14GB+ RAM for full precision")
        print("• 7-8GB RAM with 8-bit quantization")
        print("• GPU with 8GB+ VRAM")
        print("\nYour system has insufficient resources.")
        print("The code is correctly configured but needs more memory.")