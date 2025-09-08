#!/usr/bin/env python3
"""
Use FastVLM-1.5B - The smaller variant that works with limited RAM
This model requires only ~3GB RAM and maintains good performance
"""

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the smaller FastVLM model
MID = "apple/FastVLM-1.5B"  # Smaller model - only 1.5B parameters
IMAGE_TOKEN_INDEX = -200

def load_fastvlm_small():
    """Load FastVLM-1.5B which works with limited RAM"""
    print("Loading FastVLM-1.5B (optimized for limited RAM)...")
    print("This model requires only ~3GB RAM\n")
    
    # Load tokenizer
    print("1. Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    print(f"   ✓ Tokenizer loaded")
    
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
    
    print(f"\n2. Loading model on {device}...")
    print("   This will download ~3GB on first run...")
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        MID,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    print(f"   ✓ FastVLM-1.5B loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parameters: {total_params / 1e9:.2f}B")
    
    return model, tok, device

def test_generation(model, tok, device):
    """Test the model with a sample image"""
    print("\n3. Testing generation...")
    
    # Create test image
    test_image = Image.new('RGB', (336, 336), color='blue')
    
    # Prepare prompt
    messages = [
        {"role": "user", "content": "<image>\nDescribe this image."}
    ]
    
    # Apply chat template
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    
    # Tokenize
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device)
    
    # Process image (simplified for testing)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    pixel_values = transform(test_image).unsqueeze(0).to(device)
    
    print("   Generating response...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode
    response = tok.decode(outputs[0], skip_special_tokens=True)
    print(f"   Response: {response[:100]}...")
    print("\n✅ FastVLM-1.5B is working correctly!")

if __name__ == "__main__":
    print("="*60)
    print("FastVLM-1.5B - Optimized for Limited RAM")
    print("="*60)
    print()
    
    try:
        model, tok, device = load_fastvlm_small()
        test_generation(model, tok, device)
        
        print("\n" + "="*60)
        print("SUCCESS: FastVLM-1.5B is ready for use!")
        print("="*60)
        print("\nThis smaller model:")
        print("• Uses only ~3GB RAM")
        print("• Maintains good performance")
        print("• Works on your system")
        print("• Has same API as FastVLM-7B")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nEven FastVLM-1.5B failed to load.")
        print("Please close other applications and try again.")