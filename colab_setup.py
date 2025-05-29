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
    
    # Streamlit is much more stable than gradio - no compatibility issues!
    print("🎉 Installing Streamlit (stable web framework)...")
    
    # Install streamlit - much simpler and more reliable
    streamlit_commands = [
        "pip install --no-cache-dir streamlit>=1.28.0",
    ]
    
    for cmd in streamlit_commands:
        print(f"Running: {cmd}")
        run_command(cmd, ignore_errors=True)
    
    # Test streamlit import
    print("🧪 Testing streamlit import...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully!")
        print(f"✅ Streamlit version: {st.__version__}")
    except Exception as e:
        print(f"⚠️ Streamlit import failed: {e}")
        print("🔄 Trying alternative install...")
        
        # Alternative install if needed
        run_command("pip install --upgrade streamlit", ignore_errors=True)
        
        try:
            import streamlit as st
            print("✅ Streamlit fixed with alternative approach!")
            print(f"✅ Streamlit version: {st.__version__}")
        except Exception as e2:
            print(f"❌ Streamlit install failed: {e2}")
    
    # Essential packages
    essential_packages = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0", 
        "sentence-transformers>=2.2.0",
        "pdfplumber>=0.7.0",
        "PyPDF2>=2.0.0",
        "chromadb>=0.4.0",
        "python-dotenv",
        "openpyxl>=3.1.0",  # Excel XLSX support
        "xlrd>=2.0.1"       # Excel XLS support
    ]
    
    print("📦 Installing essential packages...")
    for package in essential_packages:
        print(f"Installing {package}...")
        success = run_command(f"pip install {package}", ignore_errors=True)
        if not success:
            # Try with --no-deps if regular install fails
            run_command(f"pip install --no-deps {package}", ignore_errors=True)
    
    # Optional packages
    optional_packages = [
        "openai>=1.0.0",
        "bitsandbytes",  # May fail on some systems
        "torch>=2.0.0"  # Ensure latest torch
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

def check_excel_support():
    """Check Excel processing capabilities"""
    print("📊 Checking Excel support...")
    
    try:
        import openpyxl
        print(f"✅ openpyxl version: {openpyxl.__version__}")
    except ImportError:
        print("❌ openpyxl not available - XLSX files won't work")
    
    try:
        import xlrd
        print(f"✅ xlrd version: {xlrd.__VERSION__}")
    except ImportError:
        print("❌ xlrd not available - XLS files won't work")
    
    try:
        import pandas as pd
        print(f"✅ pandas version: {pd.__version__}")
    except ImportError:
        print("❌ pandas not available - Excel processing won't work")
    
    print("📋 Supported Excel formats: .xls, .xlsx, .xlsm")

def fix_streamlit_manual():
    """Manual streamlit fix function (usually not needed)"""
    print("🔧 Manual Streamlit Fix Options (if needed):")
    print("\nStreamlit is much more stable than gradio, but if you have issues:")
    
    commands = [
        "# Option 1: Clean install",
        "!pip uninstall -y streamlit",
        "!pip install streamlit>=1.28.0",
        "",
        "# Option 2: Force latest version", 
        "!pip install --upgrade --force-reinstall streamlit",
        "",
        "# Option 3: Alternative version",
        "!pip install streamlit==1.30.0",
        "",
        "# Test command:",
        "import streamlit as st",
        "print(f'Streamlit version: {st.__version__}')"
    ]
    
    with open("streamlit_fix_commands.txt", "w") as f:
        f.write("\n".join(commands))
    
    print("📝 Commands saved to streamlit_fix_commands.txt")
    
    for cmd in commands:
        print(cmd)

def create_startup_notebook():
    """Create a startup notebook for Colab"""
    
    notebook_content = """
# PDF & Excel ChatBot - Google Colab Setup (Streamlit Version)

🆕 **NEW: Excel Support Added!** 
- ✅ XLS, XLSX, XLSM files supported
- ✅ Multi-sheet processing  
- ✅ Automatic data analysis

## 1. Initial Setup
```python
# Run the setup script (now includes Excel support!)
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

## 3. Check Excel Support (NEW!)
```python
# Verify Excel libraries are installed
try:
    import openpyxl
    import xlrd
    import pandas as pd
    print("✅ Excel support ready!")
    print(f"   openpyxl: {openpyxl.__version__}")
    print(f"   xlrd: {xlrd.__VERSION__}")
    print(f"   pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Excel support missing: {e}")
    print("Run: !pip install openpyxl xlrd")
```

## 4. HuggingFace Login (for Llama models)
```python
# Get token from: https://huggingface.co/settings/tokens
from huggingface_hub import login
login(token="your_token_here")
```

## 5. Start ChatBot with Streamlit
```python
# Method 1: Direct run (creates public tunnel automatically in Colab)
!streamlit run chat_bot.py &

# Method 2: With custom port
!streamlit run chat_bot.py --server.port 8501 &

# Method 3: Check if running
!ps aux | grep streamlit
```

## 6. Access Your App
```python
# Streamlit will automatically create a public URL in Colab
# Look for the "External URL" in the output above
# It will be something like: https://xyz.streamlit.app

# Or use ngrok for custom domain:
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")
```

## 7. Memory Management (if needed)
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

## 8. Troubleshooting

### Common Issues:
1. **Streamlit not starting**: Check if port 8501 is free
2. **Memory error**: Restart runtime, use smaller model
3. **HuggingFace error**: Check token, accept model terms
4. **Slow loading**: Use quantized models (4-bit/8-bit)
5. **Excel not loading**: Check openpyxl/xlrd installation

### Excel-Specific Issues:
- **"No module named 'openpyxl'"**: Run `!pip install openpyxl`
- **"xlrd.biffh.XLRDError"**: Old XLS format, try XLSX
- **Empty sheet error**: Check if Excel file has data
- **Encoding issues**: Save Excel as UTF-8

### Model Recommendations by GPU Memory:
- **40GB+**: Llama 3.1 70B (best quality)
- **20GB+**: Llama 3.1 8B (recommended)  
- **15GB+**: Mistral 7B (fast)
- **10GB+**: Llama 3.2 3B (light)
- **<10GB**: Use quantization (8-bit/4-bit)

### Streamlit Commands:
```python
# Stop streamlit
!pkill -f streamlit

# Restart streamlit
!streamlit run chat_bot.py --server.port 8501 &

# Check streamlit processes
!ps aux | grep streamlit
```

### File Upload Tips:
- **PDF files**: Works with text and tables
- **Excel files**: .xls, .xlsx, .xlsm supported
- **Max file size**: Usually 200MB in Colab
- **Multi-sheet Excel**: All sheets are processed automatically

### Excel Query Examples:
Once you upload an Excel file, try these queries:
- "Hangi sayfalarda hangi veriler var?" (Which data is on which sheets?)
- "En yüksek değer nedir?" (What is the highest value?)
- "Tablo verilerini özetle" (Summarize the table data)
- "Sayısal sütunları analiz et" (Analyze numeric columns)

### Advantages of Streamlit over Gradio:
- ✅ **No dependency conflicts** (major win!)
- ✅ **Better file uploads** and handling  
- ✅ **More stable** in Google Colab
- ✅ **Better session management**
- ✅ **Cleaner UI** and better UX
- ✅ **Built-in responsive design**
- ✅ **Better Excel file handling**
"""
    
    with open("colab_startup_streamlit.md", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("📝 Created colab_startup_streamlit.md with usage instructions")

def main():
    """Main setup function"""
    
    if not check_colab():
        print("⚠️ This script is optimized for Google Colab")
        print("For local installation, use requirements.txt")
    
    # Install packages with gradio fix
    install_packages()
    
    # Setup HuggingFace
    setup_huggingface()
    
    # Check GPU
    check_gpu()
    
    # Check Excel support
    check_excel_support()
    
    # Create startup guide with troubleshooting
    create_startup_notebook()
    
    # Provide manual fix options
    fix_streamlit_manual()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Check colab_startup_streamlit.md for detailed instructions")
    print("2. If streamlit import fails, check streamlit_fix_commands.txt")
    print("3. Get HuggingFace token if using Llama models")
    print("4. Run: streamlit run chat_bot.py")
    print("5. 🆕 Upload Excel files (.xls, .xlsx, .xlsm) for analysis!")
    
    print("\n📊 Excel Features:")
    print("- Multi-sheet Excel files supported")
    print("- Automatic data analysis and statistics")  
    print("- Query your Excel data with natural language")
    print("- Example: 'What is the total revenue in Q1?'")
    
    print("\n⚠️ If you STILL see streamlit errors:")
    print("- Runtime → Restart runtime")
    print("- Run: !pip install streamlit==1.28.0")
    print("- Run this script again")
    print("- Check streamlit_fix_commands.txt for more options")
    
    # Final import test
    try:
        import streamlit as st
        print("\n✅ Final check: Streamlit imported successfully!")
    except Exception as e:
        print(f"\n❌ Final check failed: {e}")
        print("📋 Use manual fix commands from streamlit_emergency_fix.txt")

if __name__ == "__main__":
    main() 