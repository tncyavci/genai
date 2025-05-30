#!/usr/bin/env python3
"""
Colab A100 Quick Commands
Copy-paste these commands to your Colab notebook for A100 optimization
"""

# 🔥 STEP 1: Install Dependencies with A100 optimizations
colab_setup_commands = """
# A100 optimized installation
!pip install --upgrade pip
!pip install llama-cpp-python --force-reinstall --no-cache-dir
!pip install streamlit sentence-transformers chromadb
!pip install pdfplumber PyPDF2 pandas openpyxl nltk rouge-score scikit-learn

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# A100 GPU optimizations
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.9)

print("✅ A100 optimized packages installed!")
"""

# 🚀 STEP 2: Mount Drive and Check GPU
drive_and_gpu_check = """
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check A100 GPU
import torch
print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Apply A100 optimizations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_CACHE_DISABLE"] = "0"

print("✅ A100 environment ready!")
"""

# 📁 STEP 3: Setup Project and Model Paths
project_setup = """
# Clone or upload your project
!rm -rf /content/genai
!git clone YOUR_REPO_URL /content/genai  # Replace with your repo
# OR upload files manually

# Change to project directory
%cd /content/genai

# Update model path in your code (update this path!)
model_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Check if model exists
import os
if os.path.exists(model_path):
    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"✅ Model found: {size_gb:.1f}GB")
else:
    print(f"❌ Model not found: {model_path}")
    print("📝 Update the path above with your actual model location")
"""

# 🧪 STEP 4: Quick A100 Test
quick_test = """
# Run A100 optimization test
!python a100_quick_setup.py

# This will:
# - Detect A100 GPU
# - Mount Drive automatically
# - Find your Mistral model
# - Test loading with A100 settings
# - Test inference speed
# - Show performance metrics
"""

# 🚀 STEP 5: Start Streamlit with A100 optimizations
streamlit_launch = """
# Install pyngrok for public URL
!pip install pyngrok

# Start Streamlit with A100 optimizations
import subprocess
import threading
import time

def run_streamlit():
    subprocess.run(["streamlit", "run", "chat_bot.py", "--server.port", "8501", "--server.headless", "true"])

# Start Streamlit in background
streamlit_thread = threading.Thread(target=run_streamlit)
streamlit_thread.start()

# Wait a moment for startup
time.sleep(10)

# Create public tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"🌐 Public URL: {public_url}")
print("🚀 A100 optimized ChatBot is ready!")
"""

# 🔧 STEP 6: Update Your Model Path (IMPORTANT!)
model_path_update = """
# IMPORTANT: Update these paths with your actual model location!

# Option 1: Edit drive_model_config.py
model_paths_to_update = {
    "current_model": "/content/drive/MyDrive/YOUR_ACTUAL_PATH/mistral-7b-instruct.gguf",
    "mistral_7b_instruct_gguf": "/content/drive/MyDrive/YOUR_ACTUAL_PATH/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
}

# Option 2: Or copy model to local storage for faster access
def copy_model_to_local(drive_path):
    import shutil
    import os
    
    if not os.path.exists(drive_path):
        print(f"❌ Model not found: {drive_path}")
        return None
    
    local_dir = "/content/models_local"
    os.makedirs(local_dir, exist_ok=True)
    
    model_name = os.path.basename(drive_path)
    local_path = os.path.join(local_dir, model_name)
    
    if not os.path.exists(local_path):
        print(f"📦 Copying model to local storage...")
        shutil.copy2(drive_path, local_path)
        print(f"✅ Model copied to: {local_path}")
    
    return local_path

# Usage:
# drive_model_path = "/content/drive/MyDrive/YOUR_PATH/mistral-7b.gguf"
# local_model_path = copy_model_to_local(drive_model_path)
"""

# 📊 STEP 7: Performance Testing Commands
performance_test = """
# Test A100 performance with different settings

# Test 1: Standard settings
standard_settings = {
    "n_ctx": 4096,
    "n_batch": 2048,
    "n_threads": 4
}

# Test 2: A100 optimized settings  
a100_settings = {
    "n_ctx": 8192,
    "n_batch": 4096,
    "n_threads": 8
}

# Test 3: Maximum A100 settings
max_a100_settings = {
    "n_ctx": 16384,  # If model supports it
    "n_batch": 8192,
    "n_threads": 12
}

print("🧪 Test these settings in your chat_bot.py")
print("📊 Compare inference speeds and adjust accordingly")
"""

# 🚨 STEP 8: Troubleshooting Commands
troubleshooting = """
# Common A100 issues and fixes

# Issue 1: Model loading fails
def fix_model_loading():
    import torch
    torch.cuda.empty_cache()
    
    # Reduce settings if needed
    reduced_settings = {
        "n_ctx": 4096,
        "n_batch": 2048,
        "n_threads": 6
    }
    print("🔧 Try with reduced settings first")

# Issue 2: Out of memory
def fix_memory_issue():
    import gc
    import torch
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)  # Reduce to 80%
    print("💾 Memory cleared and reduced allocation")

# Issue 3: Slow inference
def fix_slow_inference():
    import torch
    
    # Ensure A100 optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Check GPU utilization
    if torch.cuda.is_available():
        print(f"GPU utilization: {torch.cuda.utilization()}%")
        print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    
    print("⚡ A100 optimizations reapplied")

# Issue 4: Path not found
def fix_path_issues():
    import os
    
    # Common model locations
    common_paths = [
        "/content/drive/MyDrive/models/",
        "/content/drive/MyDrive/Colab Notebooks/",
        "/content/drive/MyDrive/",
    ]
    
    print("🔍 Searching for .gguf files...")
    for path in common_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.gguf'):
                        full_path = os.path.join(root, file)
                        size_gb = os.path.getsize(full_path) / (1024**3)
                        print(f"📁 Found: {full_path} ({size_gb:.1f}GB)")

# Run this if you have path issues
# fix_path_issues()
"""

def print_all_commands():
    """Print all commands for easy copy-paste"""
    
    print("🔥 A100 COLAB OPTIMIZATION COMMANDS")
    print("=" * 60)
    
    print("\n1️⃣ SETUP DEPENDENCIES:")
    print(colab_setup_commands)
    
    print("\n2️⃣ GPU & DRIVE CHECK:")
    print(drive_and_gpu_check)
    
    print("\n3️⃣ PROJECT SETUP:")
    print(project_setup)
    
    print("\n4️⃣ A100 TEST:")
    print(quick_test)
    
    print("\n5️⃣ STREAMLIT LAUNCH:")
    print(streamlit_launch)
    
    print("\n6️⃣ MODEL PATH UPDATE:")
    print(model_path_update)
    
    print("\n7️⃣ PERFORMANCE TESTING:")
    print(performance_test)
    
    print("\n8️⃣ TROUBLESHOOTING:")
    print(troubleshooting)

if __name__ == "__main__":
    print_all_commands() 