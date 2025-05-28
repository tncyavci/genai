#!/usr/bin/env python3
"""
Streamlit App Launcher for PDF ChatBot
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'transformers', 'sentence_transformers']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements_colab.txt")
        return False
    
    print("✅ All required packages found")
    return True

def main():
    """Launch Streamlit app"""
    print("🚀 Starting PDF ChatBot with Streamlit...")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if streamlit app exists
    if not os.path.exists("chat_bot_streamlit.py"):
        print("❌ chat_bot_streamlit.py not found!")
        return
    
    # Launch streamlit
    try:
        print("🌐 Starting Streamlit server...")
        print("📄 App: chat_bot_streamlit.py")
        print("🔗 URL: http://localhost:8501")
        print("\n" + "="*50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "chat_bot_streamlit.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    main() 