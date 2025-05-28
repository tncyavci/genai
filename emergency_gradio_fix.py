#!/usr/bin/env python3
"""
Emergency Gradio Fix for Google Colab
Fixes the 'handle_file' import error from gradio_client
"""

import subprocess
import sys

def run_cmd(cmd):
    """Run command and show output"""
    print(f"🔄 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"📝 Output: {result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f"⚠️ Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def emergency_fix():
    """Emergency fix for gradio handle_file error"""
    print("🚨 EMERGENCY GRADIO FIX for handle_file error")
    print("=" * 50)
    
    # Method 1: Ultra stable old versions
    print("\n🔧 Method 1: Installing ultra-stable old versions...")
    commands_1 = [
        "pip uninstall -y gradio gradio-client",
        "pip install --no-cache-dir gradio==4.16.0",
        "pip install --no-cache-dir gradio-client==0.8.1"
    ]
    
    for cmd in commands_1:
        run_cmd(cmd)
    
    # Test Method 1
    try:
        import gradio as gr
        print(f"✅ SUCCESS! Gradio {gr.__version__} working!")
        return True
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Alternative stable versions
    print("\n🔧 Method 2: Alternative stable versions...")
    commands_2 = [
        "pip uninstall -y gradio gradio-client",
        "pip install --no-cache-dir gradio==4.20.0",
        "pip install --no-cache-dir gradio-client==0.10.0"
    ]
    
    for cmd in commands_2:
        run_cmd(cmd)
    
    # Test Method 2
    try:
        import gradio as gr
        print(f"✅ SUCCESS! Gradio {gr.__version__} working!")
        return True
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Client downgrade only
    print("\n🔧 Method 3: Keep gradio, downgrade client only...")
    commands_3 = [
        "pip uninstall -y gradio-client",
        "pip install gradio-client==0.8.1"
    ]
    
    for cmd in commands_3:
        run_cmd(cmd)
    
    # Test Method 3
    try:
        import gradio as gr
        print(f"✅ SUCCESS! Gradio {gr.__version__} working!")
        return True
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    # All methods failed
    print("\n💀 All methods failed!")
    print("📋 Manual steps to try:")
    print("1. Runtime → Restart runtime")
    print("2. !pip install gradio==4.16.0")
    print("3. Test: import gradio as gr")
    print("4. If still fails, try Streamlit instead")
    
    return False

if __name__ == "__main__":
    emergency_fix() 