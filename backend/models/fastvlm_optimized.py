"""
FastVLM-7B Optimized Implementation for Limited RAM
Uses multiple optimization techniques to run on systems with <8GB RAM
"""

import os
import gc
import torch
import psutil
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# FastVLM constants
IMAGE_TOKEN_INDEX = -200
MID = "apple/FastVLM-7B"

class OptimizedFastVLM:
    """Memory-optimized FastVLM-7B implementation"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = self._get_device()
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        
    def _get_device(self):
        """Determine best device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_available_memory(self):
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / 1e9
    
    def _optimize_memory_usage(self):
        """Aggressively optimize memory usage"""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch caches
        if self.device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Set memory growth settings
        if self.device == "mps":
            torch.mps.set_per_process_memory_fraction(0.0)
    
    def load_model_optimized(self):
        """Load FastVLM-7B with aggressive memory optimizations"""
        available_gb = self._get_available_memory()
        print(f"\nOptimized FastVLM-7B Loading")
        print(f"Available memory: {available_gb:.2f} GB")
        print(f"Device: {self.device}")
        
        # Step 1: Load tokenizer (minimal memory)
        print("\n1. Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MID,
            trust_remote_code=True
        )
        print(f"   ✓ Tokenizer loaded")
        
        # Step 2: Load config to understand model structure
        print("\n2. Loading model configuration...")
        self.config = AutoConfig.from_pretrained(
            MID,
            trust_remote_code=True
        )
        print(f"   ✓ Config loaded")
        
        # Step 3: Determine optimization strategy based on available memory
        if available_gb < 6:
            print("\n3. Using EXTREME optimization (<6GB RAM)")
            return self._load_with_extreme_optimization()
        elif available_gb < 10:
            print("\n3. Using HIGH optimization (6-10GB RAM)")
            return self._load_with_high_optimization()
        else:
            print("\n3. Using STANDARD optimization (10GB+ RAM)")
            return self._load_with_standard_optimization()
    
    def _load_with_extreme_optimization(self):
        """Load with extreme optimizations for <6GB RAM"""
        try:
            print("   Strategy: Dynamic quantization + memory mapping")
            
            # First try: Load in int8 without bitsandbytes
            try:
                print("   Attempting dynamic int8 quantization...")
                
                # Load model in float16 first
                self.model = AutoModelForCausalLM.from_pretrained(
                    MID,
                    torch_dtype=torch.int8 if self.device == "cpu" else torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                # Apply dynamic quantization for CPU
                if self.device == "cpu":
                    import torch.quantization as quant
                    self.model = quant.quantize_dynamic(
                        self.model, 
                        {torch.nn.Linear}, 
                        dtype=torch.qint8
                    )
                    print("   ✓ Applied dynamic int8 quantization")
                else:
                    # For MPS, use float16 and aggressive memory clearing
                    self._optimize_memory_usage()
                    self.model = self.model.to(self.device)
                    print("   ✓ Loaded with float16 and memory optimization")
                
                return True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   Standard loading failed: Out of memory")
                else:
                    print(f"   Standard loading failed: {e}")
                
            # Fallback: Try with even more aggressive settings
            print("   Fallback: Loading with maximum memory savings...")
            
            # Set memory fraction for MPS
            if self.device == "mps":
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
            
            # Load with minimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                MID,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=False,  # Disable KV cache
            )
            
            # Manually optimize each layer
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Convert to half precision
                    module.half()
                    # Clear gradients
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad = False
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = False
            
            print("   ✓ Loaded with maximum memory optimization")
            return True
            
        except Exception as e:
            print(f"   ✗ Extreme optimization failed: {e}")
            return False
    
    def _load_with_high_optimization(self):
        """Load with high optimizations for 6-10GB RAM"""
        try:
            print("   Strategy: 8-bit quantization + memory mapping")
            
            # Clear memory before loading
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Load with 8-bit if possible
            try:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.dtype,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    MID,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print("   ✓ Loaded with 8-bit quantization")
                return True
                
            except (ImportError, RuntimeError):
                pass
            
            # Fallback: Load with dtype optimization
            print("   Fallback: Loading with float16 precision")
            self.model = AutoModelForCausalLM.from_pretrained(
                MID,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move to device in chunks to avoid memory spike
            if self.device != "cpu":
                self.model = self._move_to_device_in_chunks(self.model)
            
            print("   ✓ Loaded with float16 precision")
            return True
            
        except Exception as e:
            print(f"   ✗ High optimization failed: {e}")
            return False
    
    def _load_with_standard_optimization(self):
        """Load with standard optimizations for 10GB+ RAM"""
        try:
            print("   Strategy: Standard float16 with memory mapping")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MID,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            print("   ✓ Loaded with standard optimization")
            return True
            
        except Exception as e:
            print(f"   ✗ Standard optimization failed: {e}")
            return False
    
    def _load_with_manual_splitting(self):
        """Manually split model across devices"""
        try:
            print("   Loading model in parts...")
            
            # Load model with init_empty_weights to avoid memory usage
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config,
                    trust_remote_code=True
                )
            
            # Create device map for splitting
            device_map = self._create_device_map()
            
            # Load and dispatch
            self.model = load_checkpoint_and_dispatch(
                self.model,
                MID,
                device_map=device_map,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            
            print("   ✓ Model loaded with manual splitting")
            return True
            
        except Exception as e:
            print(f"   ✗ Manual splitting failed: {e}")
            return False
    
    def _create_device_map(self):
        """Create optimal device map for model splitting"""
        # Split model layers across available devices
        if self.device == "mps":
            # Put embedding and first layers on MPS, rest on CPU
            return {
                "model.embed_tokens": "mps",
                "model.layers.0": "mps",
                "model.layers.1": "mps",
                "model.layers.2": "mps",
                "model.layers.3": "mps",
                "model.layers.4": "cpu",
                "model.layers.5": "cpu",
                "model.layers.6": "cpu",
                "model.layers.7": "cpu",
                "model.norm": "cpu",
                "lm_head": "cpu",
            }
        else:
            return "auto"
    
    def _move_to_device_in_chunks(self, model):
        """Move model to device in chunks to avoid memory spikes"""
        print("   Moving model to device in chunks...")
        
        # Move parameters one by one
        for name, param in model.named_parameters():
            param.data = param.data.to(self.device)
            if "." in name and name.count(".") % 5 == 0:
                # Garbage collect every few layers
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
        
        return model
    
    def optimize_for_inference(self):
        """Apply inference-time optimizations"""
        if self.model is None:
            return
        
        print("\n4. Applying inference optimizations...")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("   ✓ Gradient checkpointing enabled")
        
        # Set to eval mode
        self.model.eval()
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        print("   ✓ Inference mode enabled")
        
        # Clear cache
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Report final memory usage
        final_memory = self._get_available_memory()
        print(f"\n5. Optimization complete!")
        print(f"   Final available memory: {final_memory:.2f} GB")
    
    def generate_optimized(self, image: Image.Image, prompt: str = None) -> str:
        """Memory-optimized generation"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded"
        
        # Default prompt
        if prompt is None:
            prompt = "<image>\nDescribe this image in detail."
        
        # Prepare input with minimal memory usage
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Split and tokenize
        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        
        # Process image efficiently
        if hasattr(self.model, 'get_vision_tower'):
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, 'image_processor'):
                px = vision_tower.image_processor(
                    images=image.convert("RGB"), 
                    return_tensors="pt"
                )["pixel_values"]
            else:
                # Manual processing
                px = self._process_image_minimal(image)
        else:
            px = self._process_image_minimal(image)
        
        # Move to device carefully
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        
        input_ids = input_ids.to(device)
        px = px.to(device, dtype=self.dtype)
        
        # Generate with minimal memory
        with torch.no_grad():
            # Use memory-efficient generation settings
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=px,
                max_new_tokens=256,  # Reduced for memory
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=False,  # Disable KV cache to save memory
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        del input_ids, px, outputs
        gc.collect()
        
        return response
    
    def _process_image_minimal(self, image: Image.Image) -> torch.Tensor:
        """Minimal image processing for memory efficiency"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        return transform(image).unsqueeze(0)

def test_optimized_loading():
    """Test the optimized FastVLM loading"""
    print("="*60)
    print("FastVLM-7B Optimized Loading Test")
    print("="*60)
    
    model = OptimizedFastVLM()
    
    # Try to load with optimizations
    success = model.load_model_optimized()
    
    if success:
        # Apply inference optimizations
        model.optimize_for_inference()
        
        print("\n✅ SUCCESS: FastVLM-7B loaded with optimizations!")
        print(f"   Device: {model.device}")
        print(f"   Dtype: {model.dtype}")
        
        # Test generation
        print("\n6. Testing generation...")
        test_image = Image.new('RGB', (336, 336), color='blue')
        try:
            response = model.generate_optimized(test_image)
            print(f"   ✓ Generation successful")
            print(f"   Response: {response[:100]}...")
        except Exception as e:
            print(f"   ✗ Generation failed: {e}")
    else:
        print("\n✗ Failed to load FastVLM-7B even with optimizations")
        print("\nFinal recommendations:")
        print("1. Close ALL other applications")
        print("2. Restart your computer and try again")
        print("3. Use FastVLM-1.5B instead (3GB requirement)")
        print("4. Use cloud GPU services")

if __name__ == "__main__":
    test_optimized_loading()