#!/usr/bin/env python3
"""
Simple A100 Startup Script
Just the essentials for A100 optimization
"""

import torch
import os

def setup_a100():
    """Simple A100 setup - no complications"""
    print("🔥 A100 Quick Setup")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        
        if "A100" in gpu_name:
            print("🎯 A100 detected - applying optimizations...")
            
            # Simple A100 optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            print("✅ A100 optimizations applied!")
            return True
        else:
            print(f"⚠️ Not A100: {gpu_name}")
            return False
    else:
        print("❌ No GPU found")
        return False

def check_model_path():
    """Check your model path"""
    # Bu path'i kendi model path'inle değiştir
    model_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    print(f"\n📁 Checking model: {model_path}")
    
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"✅ Model found: {size_gb:.1f}GB")
        return model_path
    else:
        print("❌ Model not found")
        print("📝 Update model_path in this script")
        return None

if __name__ == "__main__":
    # Simple setup
    if setup_a100():
        model_path = check_model_path()
        if model_path:
            print(f"\n🚀 Ready! Start with:")
            print(f"streamlit run chat_bot.py")
            print(f"\n📝 In ChatBot UI:")
            print(f"1. ✅ Check 'GGUF Model Kullan'")
            print(f"2. 📁 Enter path: {model_path}")
            print(f"3. 🚀 Click 'GGUF ChatBot'u Başlat'")
        else:
            print("❌ Fix model path first")
    else:
        print("❌ A100 not available") 