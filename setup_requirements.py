#!/usr/bin/env python3
"""
PDF ChatBot - Requirements Setup Script
Installs all necessary libraries for PDF processing and chatbot functionality
"""

import subprocess
import sys

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed: {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install: {package}")

def main():
    """Install all required packages"""
    packages = [
        # PDF Processing
        "pypdf2",
        "pdfplumber", 
        "tabula-py",
        "camelot-py[cv]",
        
        # Text Processing & Embeddings
        "langchain",
        "langchain-community", 
        "sentence-transformers",
        "transformers",
        
        # Vector Databases
        "chromadb",
        "faiss-cpu",
        
        # LLM APIs
        "openai",
        "anthropic",
        
        # Data Processing
        "pandas",
        "numpy",
        "openpyxl",
        
        # UI Components
        "gradio",
        "streamlit",
        
        # Utilities
        "python-dotenv",
        "tqdm"
    ]
    
    print("🚀 Installing PDF ChatBot requirements...")
    print(f"📦 Total packages to install: {len(packages)}")
    
    for package in packages:
        install_package(package)
    
    print("\n✅ Setup completed! Ready to build PDF ChatBot.")

if __name__ == "__main__":
    main() 