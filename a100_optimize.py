#!/usr/bin/env python3
"""
A100 Optimization - Mevcut workflow'a eklemek için
"""

import torch
import os

def apply_a100_optimizations():
    """A100 GPU optimizasyonlarını uygula"""
    print("🔥 A100 GPU Optimizasyonları")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ GPU bulunamadı!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_memory:.1f}GB")
    
    # A100 için özel optimizasyonlar
    if "A100" in gpu_name:
        print("🎯 A100 DETECTED - Maximum optimizations!")
        
        # PyTorch optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["CUDA_CACHE_DISABLE"] = "0"
        
        # HuggingFace cache
        os.environ["HF_HOME"] = "/content/hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/content/hf_cache"
        
        print("⚡ cuDNN benchmark: ON")
        print("⚡ TF32 matmul: ON") 
        print("⚡ GPU memory: 90%")
        print("⚡ CUDA optimizations: ON")
        print("✅ A100 optimizations applied!")
        
        return True
    else:
        print(f"⚠️ GPU: {gpu_name} (Not A100)")
        print("📝 Basic optimizations applied")
        
        # Basic optimizasyonlar
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        return True

def check_model_path():
    """Model path kontrolü"""
    # Kendi model path'inizi buraya yazın
    model_paths = [
        "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "/content/drive/MyDrive/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "/content/drive/MyDrive/mistral-7b.gguf"
    ]
    
    print("\n📁 Model path kontrolü:")
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            size_gb = os.path.getsize(model_path) / (1024**3)
            print(f"✅ Found: {os.path.basename(model_path)} ({size_gb:.1f}GB)")
            print(f"📍 Path: {model_path}")
            return model_path
    
    print("❌ Hiçbir model bulunamadı!")
    print("📝 Model path'lerini güncelle:")
    for path in model_paths:
        print(f"   - {path}")
    
    return None

if __name__ == "__main__":
    # A100 optimizasyonlarını uygula
    success = apply_a100_optimizations()
    
    if success:
        model_path = check_model_path()
        
        print("\n🚀 Optimizasyonlar tamamlandı!")
        print("📋 Sıradaki adımlar:")
        print("1. Streamlit'i başlat (mevcut workflow)")
        print("2. ngrok tunnel oluştur (mevcut workflow)")
        
        if model_path:
            print(f"\n💡 ChatBot UI'da kullan:")
            print(f"✅ GGUF Model Kullan: ✓")
            print(f"📁 Model path: {model_path}")
        else:
            print("\n⚠️ Model path'ini manuel gir!")
    else:
        print("❌ Optimizasyonlar başarısız!") 