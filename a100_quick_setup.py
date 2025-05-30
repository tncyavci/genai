#!/usr/bin/env python3
"""
A100 Quick Setup and Test Script
Optimized for NVIDIA A100 40GB GPU with Mistral 7B
"""

import os
import time
import torch
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_a100_setup():
    """Comprehensive A100 setup check"""
    print("🔥 A100 GPU Quick Setup & Test")
    print("=" * 50)
    
    # 1. GPU Check
    print("\n1️⃣ GPU Detection:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ✅ GPU: {gpu_name}")
        print(f"   💾 Memory: {gpu_memory:.1f}GB")
        
        if "A100" in gpu_name:
            print("   🎯 A100 DETECTED - OPTIMAL!")
        else:
            print(f"   ⚠️ Non-A100 GPU: {gpu_name}")
    else:
        print("   ❌ No GPU detected")
        return False
    
    # 2. Drive Mount
    print("\n2️⃣ Google Drive:")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("   📁 Mounting Drive...")
            drive.mount('/content/drive')
        print("   ✅ Drive mounted")
    except Exception as e:
        print(f"   ❌ Drive mount failed: {e}")
        return False
    
    # 3. Model Path Check
    print("\n3️⃣ Model Path Check:")
    common_paths = [
        "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q8_0.gguf",
        "/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "/content/drive/MyDrive/mistral-7b.gguf",
    ]
    
    found_model = None
    for path in common_paths:
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"   ✅ Found: {os.path.basename(path)} ({size_gb:.1f}GB)")
            found_model = path
            break
        else:
            print(f"   ❌ Not found: {os.path.basename(path)}")
    
    if not found_model:
        print("\n   📝 To fix:")
        print("   1. Upload your Mistral 7B GGUF model to Drive")
        print("   2. Update path in drive_model_config.py")
        print("   3. Recommended: Use Q4_K_M or Q8_0 quantization")
        return False
    
    return found_model

def test_a100_optimizations():
    """Test A100 specific optimizations"""
    print("\n4️⃣ A100 Optimizations:")
    
    # Set A100 optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    print("   ✅ cuDNN benchmark enabled")
    print("   ✅ TF32 enabled for A100")
    print("   ✅ GPU memory fraction: 90%")
    print("   ✅ CUDA optimizations applied")
    
    # Memory test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   📊 Memory allocated: {allocated:.2f}GB")
        print(f"   📊 Memory reserved: {reserved:.2f}GB")

def test_model_loading(model_path):
    """Test GGUF model loading with A100 settings"""
    print("\n5️⃣ Model Loading Test:")
    
    try:
        from llama_cpp import Llama
        
        start_time = time.time()
        
        # A100 optimized settings
        model = Llama(
            model_path=model_path,
            n_ctx=8192,           # Large context for A100
            n_gpu_layers=-1,      # All layers to GPU
            n_batch=4096,         # Large batch size
            n_threads=8,          # More CPU threads
            verbose=False,
            use_mmap=True,
            f16_kv=True,
            low_vram=False,       # A100 has plenty VRAM
        )
        
        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.1f}s")
        print(f"   🎯 Context window: 8192")
        print(f"   🔢 Batch size: 4096")
        print(f"   🧵 CPU threads: 8")
        
        # Quick inference test
        print("\n6️⃣ Inference Speed Test:")
        test_prompt = "Finansal rapor nedir?"
        
        start_time = time.time()
        response = model(
            test_prompt,
            max_tokens=50,
            temperature=0.7,
            stop=["</s>"],
            echo=False
        )
        inference_time = time.time() - start_time
        
        if isinstance(response, dict) and 'choices' in response:
            tokens = len(response['choices'][0]['text'].split())
            tokens_per_sec = tokens / inference_time
            print(f"   ✅ Generated {tokens} tokens in {inference_time:.2f}s")
            print(f"   🚀 Speed: {tokens_per_sec:.1f} tokens/second")
            
            if tokens_per_sec > 20:
                print("   🔥 EXCELLENT A100 performance!")
            elif tokens_per_sec > 10:
                print("   ✅ Good performance")
            else:
                print("   ⚠️ Performance could be better")
        
        return True
        
    except ImportError:
        print("   ❌ llama-cpp-python not installed")
        print("   📝 Run: !pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def main():
    """Main A100 setup test"""
    print("🔥 A100 GPU OPTIMIZATION TEST")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Basic setup check
    model_path = check_a100_setup()
    if not model_path:
        print("\n❌ Setup incomplete - fix issues above")
        return
    
    # Step 2: Apply A100 optimizations
    test_a100_optimizations()
    
    # Step 3: Test model loading and inference
    if test_model_loading(model_path):
        print("\n🎉 A100 SETUP COMPLETE!")
        print("\n📋 Next steps:")
        print("1. Run: streamlit run chat_bot.py")
        print("2. Upload PDF files for processing")
        print("3. Enjoy high-speed inference! 🚀")
        
        print(f"\n🔧 Your optimal model: {os.path.basename(model_path)}")
        print("💡 For maximum quality, use Q8_0 quantization")
        print("⚡ For maximum speed, use Q4_K_M quantization")
    else:
        print("\n❌ Model test failed")
        print("📝 Check your model path and format")

if __name__ == "__main__":
    main() 