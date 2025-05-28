#!/usr/bin/env python3
"""
Google Colab Setup Script for PDF ChatBot
Handles dependency conflicts and optimized installation
"""

import subprocess
import sys
import os

def run_command(command, ignore_errors=False):
    """Run command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and not ignore_errors:
            print(f"⚠️ Warning: {command}")
            print(f"Error: {result.stderr}")
        else:
            print(f"✅ {command}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed: {command} - {e}")
        return False

def check_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_packages():
    """Install packages with proper handling"""
    
    print("🚀 Starting Google Colab setup for PDF ChatBot...")
    
    # Essential packages first
    essential_packages = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0", 
        "sentence-transformers>=2.2.0",
        "pdfplumber>=0.7.0",
        "PyPDF2>=2.0.0",
        "chromadb>=0.4.0",
        "gradio>=4.0.0",
        "python-dotenv"
    ]
    
    print("📦 Installing essential packages...")
    for package in essential_packages:
        print(f"Installing {package}...")
        success = run_command(f"pip install --no-deps {package}", ignore_errors=True)
        if not success:
            # Try without --no-deps
            run_command(f"pip install {package}", ignore_errors=True)
    
    # Optional packages
    optional_packages = [
        "openai>=1.0.0",
        "bitsandbytes"  # May fail on some systems
    ]
    
    print("🔧 Installing optional packages...")
    for package in optional_packages:
        print(f"Trying to install {package}...")
        run_command(f"pip install {package}", ignore_errors=True)

def setup_huggingface():
    """Setup HuggingFace cache and login info"""
    print("🤗 Setting up HuggingFace...")
    
    # Set cache directory based on environment
    if check_colab():
        cache_dir = "/content/hf_cache"
    else:
        cache_dir = "./hf_cache"
    
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    print(f"📁 HuggingFace cache set to: {cache_dir}")
    
    # Create cache directory
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✅ Cache directory created: {cache_dir}")
    except OSError as e:
        print(f"⚠️ Could not create cache directory: {e}")
        print("Using default cache location")
    
    print("💡 To use gated models (like Llama), run:")
    print("   from huggingface_hub import login")
    print("   login(token='your_token_here')")
    print("   Get token from: https://huggingface.co/settings/tokens")

def check_gpu():
    """Check GPU availability"""
    import torch
    
    print("🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f}GB")
        
        # Recommend model based on memory
        if gpu_memory > 20:
            print("🎯 Recommended: Llama 3.1 8B (16GB needed)")
        elif gpu_memory > 15:
            print("🎯 Recommended: Mistral 7B (14GB needed)")
        else:
            print("🎯 Recommended: Smaller models or quantization")
    else:
        print("⚠️ No GPU detected - Local LLM will be very slow")

def create_startup_notebook():
    """Create a startup notebook for Colab"""
    
    notebook_content = """
# PDF ChatBot - Google Colab Setup

## 1. Check System
```python
!python colab_setup.py
```

## 2. Check GPU (Important!)
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

## 3. HuggingFace Login (for Llama models)
```python
# Get token from: https://huggingface.co/settings/tokens
from huggingface_hub import login
login(token="your_token_here")
```

## 4. Start ChatBot
```python
# Start with public URL for Colab
!python chat_bot.py &

# Or programmatically:
from chat_bot import create_gradio_interface
demo = create_gradio_interface()
demo.launch(share=True, debug=False)  # share=True creates public URL
```

## 5. Memory Management (if needed)
```python
import gc
import torch

# Clear memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Check memory usage
if torch.cuda.is_available():
    print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
```

## Model Recommendations by GPU Memory:

- **40GB+**: Llama 3.1 70B (best quality)
- **20GB+**: Llama 3.1 8B (recommended)  
- **15GB+**: Mistral 7B (fast)
- **10GB+**: Llama 3.2 3B (light)
- **<10GB**: Use quantization (8-bit/4-bit)
"""
    
    with open("colab_startup.md", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("📝 Created colab_startup.md with usage instructions")

def main():
    """Main setup function"""
    
    if not check_colab():
        print("⚠️ This script is optimized for Google Colab")
        print("For local installation, use requirements.txt")
    
    # Install packages
    install_packages()
    
    # Setup HuggingFace
    setup_huggingface()
    
    # Check GPU
    check_gpu()
    
    # Create startup guide
    create_startup_notebook()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Check colab_startup.md for usage instructions")
    print("2. Get HuggingFace token if using Llama models")
    print("3. Run: python chat_bot.py")
    print("4. Access the interface with share=True for public URL")
    
    print("\n⚠️ If you see dependency conflicts:")
    print("- Restart runtime (Runtime → Restart runtime)")
    print("- Run this script again")

if __name__ == "__main__":
    main() 