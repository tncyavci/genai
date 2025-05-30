#!/usr/bin/env python3
"""
Google Drive Local Model Configuration - A100 Optimized
For using custom Mistral 7B model from Drive with A100 GPU optimizations
"""

import os
import logging
import torch
from llm_service_local import LocalLLMService

logger = logging.getLogger(__name__)

def setup_a100_environment():
    """Setup A100 GPU environment for maximum performance"""
    
    # A100 GPU optimizations
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async GPU operations
    os.environ["CUDA_CACHE_DISABLE"] = "0"    # Enable CUDA cache
    
    # HuggingFace cache settings
    os.environ["HF_HOME"] = "/content/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/content/hf_cache"
    
    # PyTorch settings for A100
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
        
        # Check GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🚀 GPU: {gpu_name}")
        logger.info(f"💾 Memory: {gpu_memory:.1f}GB")
        
        if "A100" in gpu_name:
            logger.info("✅ A100 detected - Using optimal settings")
        else:
            logger.warning(f"⚠️ Non-A100 GPU detected: {gpu_name}")
    
    # Create cache directory
    os.makedirs("/content/hf_cache", exist_ok=True)
    
    logger.info("✅ A100 environment setup complete")

def create_drive_model_service(model_path: str, model_type: str = "mistral") -> LocalLLMService:
    """
    Create LLM service using model from Google Drive - A100 Optimized
    
    Args:
        model_path: Path to model in Drive (e.g., "/content/drive/MyDrive/models/mistral-7b-instruct.gguf")
        model_type: Type of model ("mistral", "llama", "generic")
    """
    
    # Setup A100 environment first
    setup_a100_environment()
    
    # Verify model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Check if it's a GGUF file
    if model_path.endswith('.gguf'):
        logger.info(f"🎯 GGUF model detected: {os.path.basename(model_path)}")
        file_size = os.path.getsize(model_path) / (1024**3)  # GB
        logger.info(f"📦 Model size: {file_size:.1f}GB")
    
    logger.info(f"🚀 Loading model from Drive: {model_path}")
    logger.info(f"📁 Model type: {model_type}")
    
    # A100 optimized settings
    service = LocalLLMService(
        model_name=model_path,
        max_new_tokens=1024,  # Increased for A100
        temperature=0.7,
        # A100 specific optimizations will be handled in chat_bot.py
    )
    
    return service

def mount_drive():
    """Mount Google Drive if not already mounted"""
    try:
        from google.colab import drive
        
        if not os.path.exists('/content/drive'):
            logger.info("📁 Mounting Google Drive...")
            drive.mount('/content/drive')
            logger.info("✅ Google Drive mounted successfully")
        else:
            logger.info("✅ Google Drive already mounted")
            
        return True
    except ImportError:
        logger.warning("⚠️ Not running in Google Colab - Drive mount skipped")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to mount Drive: {e}")
        return False

# A100 Optimized Drive model paths
DRIVE_MODEL_PATHS = {
    # GGUF Models (Recommended for A100)
    "mistral_7b_instruct_gguf": "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "mistral_7b_instruct_q8": "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q8_0.gguf", 
    "mistral_7b_instruct_q5": "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    
    # HuggingFace Format Models
    "mistral_7b_hf": "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1",
    "mistral_7b_custom": "/content/drive/MyDrive/models/your-custom-mistral",
    
    # Alternative paths (update these with your actual paths)
    "current_model": "/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # UPDATE THIS
}

def get_optimal_model_path():
    """Get the best available model path for A100"""
    
    # Mount drive first
    if not mount_drive():
        logger.error("❌ Cannot access Drive models")
        return None
    
    # Check available models in order of preference for A100
    model_priority = [
        "mistral_7b_instruct_q8",   # Best quality for A100
        "mistral_7b_instruct_q5",   # Good balance
        "mistral_7b_instruct_gguf", # Most compatible
        "current_model",            # User specified
        "mistral_7b_hf",           # HuggingFace format
    ]
    
    for model_key in model_priority:
        model_path = DRIVE_MODEL_PATHS.get(model_key)
        if model_path and os.path.exists(model_path):
            logger.info(f"✅ Found model: {model_key} at {model_path}")
            return model_path
    
    logger.error("❌ No valid model found in Drive")
    logger.info("📝 Available paths to check:")
    for key, path in DRIVE_MODEL_PATHS.items():
        exists = "✅" if os.path.exists(path) else "❌"
        logger.info(f"   {exists} {key}: {path}")
    
    return None

if __name__ == "__main__":
    # Test A100 optimized drive model loading
    print("🧪 Testing A100 Optimized Drive Model Service...")
    
    # Mount drive and get optimal model
    optimal_path = get_optimal_model_path()
    
    if optimal_path:
        try:
            setup_a100_environment()
            service = create_drive_model_service(optimal_path, "mistral")
            print("✅ A100 optimized Drive model service created successfully!")
            print(f"📍 Model path: {optimal_path}")
        except Exception as e:
            print(f"❌ Failed to create service: {e}")
    else:
        print("❌ No valid model found")
        print("\n📝 To fix this:")
        print("1. Upload your Mistral 7B model to Google Drive")
        print("2. Update DRIVE_MODEL_PATHS['current_model'] with correct path")
        print("3. Recommended: Use GGUF format for A100 (Q4_K_M or Q8_0)") 