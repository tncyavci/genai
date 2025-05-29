#!/usr/bin/env python3
"""
Google Drive Local Model Configuration
For using custom Mistral 7B model from Drive
"""

import os
import logging
from llm_service_local import LocalLLMService

logger = logging.getLogger(__name__)

def create_drive_model_service(model_path: str, model_type: str = "mistral") -> LocalLLMService:
    """
    Create LLM service using model from Google Drive
    
    Args:
        model_path: Path to model in Drive (e.g., "/content/drive/MyDrive/mistral-7b")
        model_type: Type of model ("mistral", "llama", "generic")
    """
    
    # Verify model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Check required files
    required_files = ["config.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            logger.warning(f"Required file missing: {file}")
    
    logger.info(f"🚀 Loading model from Drive: {model_path}")
    logger.info(f"📁 Model type: {model_type}")
    
    # Create service with local path
    service = LocalLLMService(
        model_name=model_path,  # Use local path instead of HF model name
        max_new_tokens=512,
        temperature=0.7
    )
    
    return service

def setup_drive_environment():
    """Setup environment for Drive model usage"""
    
    # Set cache directories to avoid conflicts
    os.environ["HF_HOME"] = "/content/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/content/hf_cache"
    
    # Create cache directory
    os.makedirs("/content/hf_cache", exist_ok=True)
    
    logger.info("✅ Drive environment setup complete")

# Common Drive model paths (update these with your actual paths)
DRIVE_MODEL_PATHS = {
    "mistral_7b": "/content/drive/MyDrive/models/mistral-7b",
    "mistral_7b_instruct": "/content/drive/MyDrive/models/mistral-7b-instruct",
    "custom_model": "/content/drive/MyDrive/your-model-folder"
}

if __name__ == "__main__":
    # Test drive model loading
    print("🧪 Testing Drive Model Service...")
    
    # Example usage
    model_path = "/content/drive/MyDrive/mistral-7b"  # Update this path
    
    if os.path.exists(model_path):
        setup_drive_environment()
        service = create_drive_model_service(model_path, "mistral")
        print("✅ Drive model service created successfully!")
    else:
        print(f"❌ Model path not found: {model_path}")
        print("📝 Update the path in this script to match your Drive structure") 