import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import io
import json
import re
from datetime import datetime
from PIL import Image
import numpy as np

# Model loading flags
TORCH_AVAILABLE = False
MODEL_LOADED = False
MODEL_TYPE = "mock"  # "fastvlm", "llava", "blip", "mock"

# FastVLM specific constants
IMAGE_TOKEN_INDEX = -200  # Special token for image placeholders in FastVLM

try:
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        AutoProcessor,
        BlipProcessor, 
        BlipForConditionalGeneration,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        BitsAndBytesConfig
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch/Transformers not fully installed: {e}")
    print("Running in mock mode - install torch and transformers for real model")

class ModelStatus:
    """Track model loading status"""
    def __init__(self):
        self.is_loaded = False
        self.model_type = "mock"
        self.model_name = None
        self.device = "cpu"
        self.error = None
        self.loading_time = None
        self.parameters_count = 0
        
    def to_dict(self):
        return {
            "is_loaded": self.is_loaded,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "device": self.device,
            "error": self.error,
            "loading_time": self.loading_time,
            "parameters_count": self.parameters_count,
            "timestamp": datetime.now().isoformat()
        }

class FastVLMModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        self.status = ModelStatus()
        self._setup_device()
        
    def _setup_device(self):
        """Setup compute device"""
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print("Using Apple Silicon MPS device")
            else:
                self.device = "cpu"
                print("Using CPU device")
        else:
            self.device = "cpu"
        self.status.device = self.device
        
    async def initialize(self, model_type: str = "auto"):
        """
        Initialize the vision-language model with fallback options.
        
        Args:
            model_type: "auto", "fastvlm", "llava", "blip", or "mock"
        """
        start_time = datetime.now()
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available - running in mock mode")
            self.status.model_type = "mock"
            self.status.error = "PyTorch not installed"
            return
        
        # Try loading models in order of preference
        if model_type == "auto":
            # Check available memory and choose appropriate model
            import psutil
            available_gb = psutil.virtual_memory().available / 1e9
            print(f"Available memory: {available_gb:.2f} GB")
            
            if available_gb < 10:
                print("Limited memory detected, prioritizing smaller models")
                models_to_try = ["fastvlm-small", "blip", "fastvlm"]
            else:
                models_to_try = ["fastvlm", "llava", "blip"]
        else:
            models_to_try = [model_type]
            
        for model_name in models_to_try:
            success = await self._try_load_model(model_name)
            if success:
                self.status.is_loaded = True
                self.status.model_type = model_name
                self.status.loading_time = (datetime.now() - start_time).total_seconds()
                print(f"Successfully loaded {model_name} model in {self.status.loading_time:.2f}s")
                return
                
        # Fallback to mock mode
        print("All model loading attempts failed - using mock mode")
        self.status.model_type = "mock"
        self.status.error = "Failed to load any vision-language model"
        
    async def _try_load_model(self, model_type: str) -> bool:
        """Try to load a specific model type"""
        try:
            print(f"Attempting to load {model_type} model...")
            
            if model_type == "fastvlm":
                return await self._load_fastvlm()
            elif model_type == "fastvlm-small":
                return await self._load_fastvlm_small()
            elif model_type == "llava":
                return await self._load_llava()
            elif model_type == "blip":
                return await self._load_blip()
            else:
                return False
                
        except Exception as e:
            print(f"Failed to load {model_type}: {e}")
            self.status.error = str(e)
            return False
            
    async def _load_fastvlm_small(self) -> bool:
        """Load smaller FastVLM variant (1.5B) for limited memory systems"""
        try:
            model_name = "apple/FastVLM-1.5B"
            print(f"Loading FastVLM-1.5B from {model_name}...")
            print("This smaller model requires ~3GB RAM and is optimized for limited memory")
            
            # Load tokenizer with trust_remote_code for Qwen2Tokenizer support
            print("Loading tokenizer with trust_remote_code=True...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Add image token to tokenizer if not present
            if not hasattr(self.tokenizer, 'IMAGE_TOKEN_INDEX'):
                self.tokenizer.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
            
            # Use float16 for memory efficiency
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            print(f"Loading model with configuration: {model_kwargs}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            self.status.model_name = model_name
            self._count_parameters()
            
            # Initialize processor for image handling
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except:
                print("Warning: Could not load processor, will use custom image processing")
                self.processor = None
            
            print(f"✓ FastVLM-1.5B loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"FastVLM-1.5B loading failed: {e}")
            return False
    
    async def _load_fastvlm(self) -> bool:
        """Load FastVLM-7B model with exact HuggingFace implementation"""
        try:
            MID = "apple/FastVLM-7B"  # Exact model ID from HuggingFace
            print(f"Loading FastVLM-7B from {MID}...")
            
            # Check available memory
            import psutil
            available_gb = psutil.virtual_memory().available / 1e9
            print(f"Available memory: {available_gb:.2f} GB")
            
            # Load tokenizer with trust_remote_code as per model card
            print("Loading tokenizer with trust_remote_code=True...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MID,
                trust_remote_code=True  # Required for Qwen2Tokenizer
            )
            
            # Set IMAGE_TOKEN_INDEX as specified in model card
            self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX  # -200
            print(f"IMAGE_TOKEN_INDEX set to {self.IMAGE_TOKEN_INDEX}")
            
            # Configure model loading - check if we can use quantization
            if available_gb < 12 and self.device == "cuda":  # Quantization only works on CUDA
                print("Implementing 8-bit quantization for memory efficiency...")
                try:
                    from transformers import BitsAndBytesConfig
                    
                    # 8-bit quantization config
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_quant_type="nf4"
                    )
                    
                    model_kwargs = {
                        "quantization_config": quantization_config,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    }
                    print("Using 8-bit quantization - model will use ~7GB RAM")
                except ImportError:
                    print("Warning: bitsandbytes not available for quantization")
                    raise RuntimeError("Insufficient memory for FastVLM-7B without quantization")
            elif available_gb < 14:
                # Try optimized loading for limited memory
                print(f"\n⚠️ Limited memory detected: {available_gb:.2f} GB")
                print("Attempting optimized loading for FastVLM-7B...")
                
                try:
                    # First try extreme optimizations
                    from models.fastvlm_extreme import ExtremeOptimizedFastVLM7B
                    
                    extreme = ExtremeOptimizedFastVLM7B()
                    if extreme.load_fastvlm_7b_extreme():
                        # Transfer to main model
                        self.model = extreme.model
                        self.tokenizer = extreme.tokenizer
                        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
                        
                        self.status.model_name = MID
                        if self.model:
                            self._count_parameters()
                        
                        print(f"✓ FastVLM-7B loaded with EXTREME optimizations!")
                        return True
                    
                    # Fallback to standard optimizations
                    from models.fastvlm_optimized import OptimizedFastVLM
                    
                    optimized = OptimizedFastVLM()
                    if optimized.load_model_optimized():
                        optimized.optimize_for_inference()
                        
                        # Transfer to main model
                        self.model = optimized.model
                        self.tokenizer = optimized.tokenizer
                        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
                        
                        self.status.model_name = MID
                        self._count_parameters()
                        
                        print(f"✓ FastVLM-7B loaded with memory optimizations!")
                        return True
                    else:
                        raise RuntimeError("Optimized loading failed")
                        
                except Exception as e:
                    print(f"\nOptimized loading failed: {e}")
                    print("\nFalling back to error message...")
                    print(f"\n⚠️ INSUFFICIENT MEMORY FOR FastVLM-7B")
                    print(f"   Available: {available_gb:.2f} GB")
                    print(f"   Required: 14GB (full) or 4-7GB (optimized)")
                    print("\nSolutions:")
                    print("1. Close other applications to free memory")
                    print("2. Use FastVLM-1.5B (smaller model)")
                    print("3. Upgrade system RAM")
                    raise RuntimeError(f"Insufficient memory: {available_gb:.2f}GB available")
            else:
                # Full precision for systems with enough RAM
                model_kwargs = {
                    "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True
                }
                print("Using full precision - model will use ~14GB RAM")
            
            print(f"Loading model with configuration: device_map=auto, trust_remote_code=True")
            self.model = AutoModelForCausalLM.from_pretrained(
                MID,
                **model_kwargs
            )
            
            self.model.eval()
            self.status.model_name = MID
            self._count_parameters()
            
            # Verify vision tower is loaded
            if hasattr(self.model, 'get_vision_tower'):
                print("✓ Vision tower (FastViTHD) loaded successfully")
            else:
                print("Warning: Vision tower not found, image processing may be limited")
            
            print(f"✓ FastVLM-7B loaded successfully with IMAGE_TOKEN_INDEX={self.IMAGE_TOKEN_INDEX}")
            print(f"✓ Model ready on {self.device} with {'8-bit quantization' if available_gb < 12 else 'full precision'}")
            return True
            
        except ImportError as e:
            if "bitsandbytes" in str(e):
                print("Error: bitsandbytes not installed. For quantization support, run:")
                print("pip install bitsandbytes")
            else:
                print(f"Import error: {e}")
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Error: Insufficient memory for FastVLM-7B")
                print("Solutions:")
                print("1. Use quantized version: apple/FastVLM-7B-int4")
                print("2. Reduce batch size")
                print("3. Use a smaller model variant (FastVLM-1.5B)")
                print("4. Add more RAM or use a GPU")
            else:
                print(f"Runtime error: {e}")
            return False
        except Exception as e:
            print(f"FastVLM loading failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
            
    async def _load_llava(self) -> bool:
        """Load LLaVA model as alternative"""
        try:
            model_name = "llava-hf/llava-1.5-7b-hf"
            
            self.processor = LlavaProcessor.from_pretrained(model_name)
            
            if self.device == "cuda":
                # Use 4-bit quantization for GPU to save memory
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                # Load in float32 for CPU
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
                
            self.model.eval()
            self.status.model_name = model_name
            self._count_parameters()
            return True
            
        except Exception as e:
            print(f"LLaVA loading failed: {e}")
            return False
            
    async def _load_blip(self) -> bool:
        """Load BLIP model as lightweight alternative"""
        try:
            model_name = "Salesforce/blip-image-captioning-large"
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            
            if self.device == "cuda":
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                ).to(self.device)
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
                
            self.model.eval()
            self.status.model_name = model_name
            self._count_parameters()
            return True
            
        except Exception as e:
            print(f"BLIP loading failed: {e}")
            return False
            
    def _count_parameters(self):
        """Count model parameters"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            self.status.parameters_count = total_params
            print(f"Model has {total_params / 1e9:.2f}B parameters")
            
    async def analyze_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze an image and return structured results.
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Check if we have a real model loaded
            if self.model is None or self.status.model_type == "mock":
                return self._mock_analysis(image)
                
            # Use appropriate analysis method based on model type
            if self.status.model_type == "fastvlm":
                return await self._analyze_with_fastvlm(image)
            elif self.status.model_type == "llava":
                return await self._analyze_with_llava(image)
            elif self.status.model_type == "blip":
                return await self._analyze_with_blip(image)
            else:
                return self._mock_analysis(image)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                "summary": f"Analysis failed: {str(e)}",
                "ui_elements": [],
                "text_snippets": [],
                "risk_flags": ["ANALYSIS_ERROR"],
                "model_info": self.status.to_dict()
            }
            
    async def _analyze_with_fastvlm(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image with FastVLM using exact HuggingFace implementation"""
        try:
            # Prepare chat message with image placeholder as per model card
            messages = [{
                "role": "user", 
                "content": """<image>\nAnalyze this screen capture and provide:
                1. A brief summary of what's visible
                2. UI elements (buttons, links, forms)
                3. Text snippets
                4. Security or privacy risks
                
                Respond in JSON format with keys: summary, ui_elements, text_snippets, risk_flags"""
            }]
            
            # Apply chat template and split around <image> token
            rendered = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            pre, post = rendered.split("<image>", 1)
            
            # Tokenize text parts separately as per model card
            pre_ids = self.tokenizer(
                pre, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids
            
            post_ids = self.tokenizer(
                post, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids
            
            # Create image token tensor with IMAGE_TOKEN_INDEX
            img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            
            # Splice tokens together: pre_text + IMAGE_TOKEN + post_text
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
            
            # Move to correct device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            
            # Process image using vision tower
            if hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
                if hasattr(vision_tower, 'image_processor'):
                    # Use the model's image processor
                    px = vision_tower.image_processor(
                        images=image.convert("RGB"), 
                        return_tensors="pt"
                    )["pixel_values"]
                    px = px.to(device, dtype=self.model.dtype)
                else:
                    # Fallback to custom processing
                    px = self._process_image_for_fastvlm(image).to(device)
            else:
                # Fallback if vision tower not available
                px = self._process_image_for_fastvlm(image).to(device)
            
            # Generate response with exact parameters from model card
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=px,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if rendered in response:
                response = response.replace(rendered, "").strip()
            
            return self._parse_model_response(response)
            
        except Exception as e:
            print(f"Error in FastVLM analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                "summary": f"Analysis failed: {str(e)}",
                "ui_elements": [],
                "text_snippets": [],
                "risk_flags": ["ANALYSIS_ERROR"],
                "model_info": self.status.to_dict(),
                "error_detail": str(e)
            }
        
    async def _analyze_with_llava(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image with LLaVA model"""
        prompt = """USER: <image>
Analyze this screen and provide a JSON response with:
- summary: what you see
- ui_elements: list of UI elements
- text_snippets: visible text
- risk_flags: any security concerns
ASSISTANT:"""
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return self._parse_model_response(response)
        
    async def _analyze_with_blip(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image with BLIP model"""
        # BLIP is primarily for captioning, so we'll use it for summary
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
            
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Since BLIP only provides captions, we'll structure it accordingly
        return {
            "summary": caption,
            "ui_elements": [],
            "text_snippets": [],
            "risk_flags": [],
            "model_info": self.status.to_dict(),
            "note": "Using BLIP model - only caption generation available"
        }
        
    def _process_image_for_model(self, image: Image.Image) -> torch.Tensor:
        """Process image for model input"""
        if not TORCH_AVAILABLE:
            return None
            
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def _process_image_for_fastvlm(self, image: Image.Image) -> torch.Tensor:
        """Process image specifically for FastVLM model"""
        if not TORCH_AVAILABLE:
            return None
        
        from torchvision import transforms
        
        # FastVLM expects 336x336 images with specific normalization
        transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
        
    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract JSON"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Ensure all required keys exist
                result = {
                    "summary": parsed.get("summary", "Analysis complete"),
                    "ui_elements": parsed.get("ui_elements", []),
                    "text_snippets": parsed.get("text_snippets", []),
                    "risk_flags": parsed.get("risk_flags", []),
                    "model_info": self.status.to_dict()
                }
                return result
        except Exception as e:
            print(f"Failed to parse model response: {e}")
            
        # Fallback: return raw response as summary
        return {
            "summary": response[:500],  # Truncate long responses
            "ui_elements": [],
            "text_snippets": [],
            "risk_flags": [],
            "model_info": self.status.to_dict(),
            "raw_response": True
        }
        
    def _mock_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Generate mock analysis for testing"""
        # Analyze image properties for more realistic mock data
        width, height = image.size
        
        # Generate mock UI elements based on image regions
        ui_elements = []
        for i in range(3):
            ui_elements.append({
                "type": ["button", "link", "input", "dropdown"][i % 4],
                "text": f"Element {i+1}",
                "position": {
                    "x": (i + 1) * width // 4,
                    "y": (i + 1) * height // 4
                }
            })
            
        return {
            "summary": f"Mock analysis of {width}x{height} screen capture. Real model not loaded.",
            "ui_elements": ui_elements,
            "text_snippets": [
                "Sample text detected",
                "Another text region",
                f"Image dimensions: {width}x{height}"
            ],
            "risk_flags": [],
            "model_info": self.status.to_dict(),
            "mock_mode": True
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return self.status.to_dict()
        
    async def reload_model(self, model_type: str = "auto") -> Dict[str, Any]:
        """Reload the model with specified type"""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.status = ModelStatus()
        self._setup_device()
        await self.initialize(model_type)
        return self.status.to_dict()