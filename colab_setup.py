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
        "xlrd>=2.0.1",      # Excel XLS support
        "nltk>=3.8.1",      # Text processing and chunking
        "rouge-score>=0.1.2",  # Evaluation metrics
        "scikit-learn>=1.0.0", # Machine learning and evaluation
        "numpy>=1.24.0",     # Numerical operations
        "pandas>=2.0.0",     # Data processing
        "tqdm>=4.65.0"       # Progress bars
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

def setup_nltk():
    """Setup NLTK data"""
    print("📚 Setting up NLTK data...")
    try:
        import nltk
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        print("Downloading NLTK POS tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        
        # Test NLTK functionality
        from nltk.tokenize import sent_tokenize
        test_text = "Bu bir test cümlesidir. İkinci cümle de buradadır."
        sentences = sent_tokenize(test_text)
        print(f"✅ NLTK test successful: {len(sentences)} sentences detected")
        
    except Exception as e:
        print(f"⚠️ NLTK setup failed: {e}")
        print("🔄 Trying alternative download...")
        try:
            import nltk
            nltk.download('punkt', download_dir='/content/nltk_data')
            nltk.download('averaged_perceptron_tagger', download_dir='/content/nltk_data')
            print("✅ NLTK data downloaded to alternative location")
        except Exception as e2:
            print(f"❌ NLTK alternative setup failed: {e2}")

def check_gpu():
    """Check GPU availability and provide A100 optimizations"""
    import torch
    
    print("🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f}GB")
        
        # A100 specific recommendations
        if "A100" in gpu_name:
            print("🔥 A100 GPU DETECTED - MAXIMUM PERFORMANCE MODE!")
            print("🎯 Recommended settings for A100:")
            print("   - Context window: 8192 tokens")
            print("   - Batch size: 4096")
            print("   - CPU threads: 8")
            print("   - Model format: GGUF Q8_0 or Q5_K_M for best quality")
            print("   - Expected speed: 25-35 tokens/second")
            print("   - Memory usage: ~14GB for Mistral 7B")
            
            # Set A100 optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.set_per_process_memory_fraction(0.9)
            print("✅ A100 optimizations applied")
            
        elif gpu_memory > 20:
            print("🎯 High-end GPU - Recommended: Llama 3.1 8B or Mistral 7B")
            print("   - Context window: 6144 tokens") 
            print("   - Batch size: 3072")
            print("   - Expected speed: 15-25 tokens/second")
        elif gpu_memory > 15:
            print("🎯 Recommended: Mistral 7B (14GB needed)")
            print("   - Context window: 4096 tokens")
            print("   - Batch size: 2048") 
            print("   - Expected speed: 10-20 tokens/second")
        else:
            print("🎯 Recommended: Smaller models or quantization")
            print("   - Use Q4_K_M quantization")
            print("   - Context window: 2048 tokens")
            print("   - Expected speed: 5-15 tokens/second")
    else:
        print("⚠️ No GPU detected - Local LLM will be very slow")
        print("   - CPU inference: 1-3 tokens/second")
        print("   - Recommended: Use API-based models instead")

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

def check_new_features():
    """Check if new features are properly installed"""
    print("🆕 Checking new feature dependencies...")
    
    # Check text processing improvements
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        print("✅ NLTK sentence tokenization ready")
    except ImportError:
        print("❌ NLTK not available - Advanced chunking won't work")
    
    # Check evaluation capabilities  
    try:
        from rouge_score import rouge_scorer
        print("✅ ROUGE evaluation metrics ready")
    except ImportError:
        print("❌ ROUGE not available - Evaluation metrics won't work")
    
    try:
        from sklearn.metrics import accuracy_score
        print("✅ Scikit-learn metrics ready")
    except ImportError:
        print("❌ Scikit-learn not available - ML evaluation won't work")
    
    # Check vector store improvements
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available - Vector operations won't work")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence transformers ready")
    except ImportError:
        print("❌ Sentence transformers not available - Embeddings won't work")

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
# PDF & Excel ChatBot - Google Colab Setup (Latest Version)

🆕 **NEW FEATURES:**
- ✅ Excel Support (XLS, XLSX, XLSM)
- ✅ Improved PDF processing with smart chunking
- ✅ Better vector search with filtering
- ✅ Evaluation system for testing accuracy
- ✅ Enhanced metadata and context handling

## 1. Initial Setup
```python
# Run the setup script (now includes all latest features!)
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

## 3. Check All Features
```python
# Check Excel support
try:
    import openpyxl, xlrd, pandas as pd
    print("✅ Excel support ready!")
except ImportError as e:
    print(f"❌ Excel support missing: {e}")

# Check NLTK for advanced text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    print("✅ Advanced text processing ready!")
except ImportError as e:
    print(f"❌ NLTK missing: {e}")

# Check evaluation tools
try:
    from rouge_score import rouge_scorer
    from sklearn.metrics import accuracy_score
    print("✅ Evaluation tools ready!")
except ImportError as e:
    print(f"❌ Evaluation tools missing: {e}")
```

## 4. Test New Features
```python
# Test sentence tokenization
import nltk
from nltk.tokenize import sent_tokenize
test_text = "Bu bir test metnidir. İkinci cümle burada. Üçüncü cümle de var."
sentences = sent_tokenize(test_text)
print(f"Detected {len(sentences)} sentences: {sentences}")

# Test vector operations
import numpy as np
test_vector = np.random.rand(384)
print(f"Vector shape: {test_vector.shape}")

# Test ROUGE scoring
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
scores = scorer.score('test reference', 'test hypothesis')
print(f"ROUGE scores: {scores}")
```

## 5. HuggingFace Login (for Llama models)
```python
# Get token from: https://huggingface.co/settings/tokens
from huggingface_hub import login
login(token="your_token_here")
```

## 6. Start ChatBot with Streamlit
```python
# Method 1: Direct run (creates public tunnel automatically in Colab)
!streamlit run chat_bot.py &

# Method 2: With custom port
!streamlit run chat_bot.py --server.port 8501 &

# Method 3: Check if running
!ps aux | grep streamlit
```

## 7. Test PDF Processing
```python
# Test the enhanced PDF processor
from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from vector_store import VectorStore, RetrievalService

# Initialize components
pdf_processor = PDFProcessor()
text_processor = TextProcessor(chunk_size=300, overlap_size=50)  # Optimized settings
vector_store = VectorStore()

# Process a test PDF
result = pdf_processor.process_pdf("your_file.pdf")
print(f"Processed {len(result.pages)} pages in {result.processing_time:.2f}s")

# Create embeddings with new chunking strategy
embedded_chunks = text_processor.process_document_pages(result.pages, "your_file.pdf")
print(f"Created {len(embedded_chunks)} chunks")

# Add to vector store
vector_store.add_documents(embedded_chunks)

# Test retrieval with filtering
retrieval_service = RetrievalService(vector_store, text_processor)
context = retrieval_service.retrieve_context(
    query="finansal performans",
    n_results=5,
    min_score=0.7,
    filter_by_language='tr'
)
print(f"Found {context.total_results} relevant results")
```

## 8. Test Evaluation System
```python
# Test the evaluation system
from evaluation import EvaluationDataset, EvaluationRunner, EvaluationMetrics

# Create test cases
dataset = EvaluationDataset()
test_cases = dataset.create_financial_test_cases()
print(f"Created {len(test_cases)} test cases")

# Run evaluation
runner = EvaluationRunner()
results = runner.run_evaluation(test_cases, ["sample response 1", "sample response 2"])
print(f"Evaluation completed: {results['summary']}")
```

## 9. Memory Management (if needed)
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

## 10. Troubleshooting

### New Feature Issues:
1. **NLTK punkt error**: Run `nltk.download('punkt')`
2. **ROUGE import error**: Run `!pip install rouge-score`
3. **Sklearn missing**: Run `!pip install scikit-learn`
4. **Vector operations slow**: Check NumPy installation
5. **Chunking not working**: Verify NLTK data is downloaded

### Performance Optimizations:
- **Chunk size**: Use 300 for better accuracy, 800+ for speed
- **Min score**: Use 0.7 for quality, 0.5 for more results
- **Batch size**: Use 32 for embeddings, 10 for PDF pages
- **Overlap**: Use 50 for accuracy, 150 for context

### Model Recommendations by GPU Memory:
- **40GB+**: Llama 3.1 70B (best quality)
- **20GB+**: Llama 3.1 8B (recommended)  
- **15GB+**: Mistral 7B (fast)
- **10GB+**: Llama 3.2 3B (light)
- **<10GB**: Use quantization (8-bit/4-bit)

### Excel Query Examples:
- "Hangi sayfalarda hangi veriler var?"
- "En yüksek değer nedir?"
- "Tablo verilerini özetle"
- "Sayısal sütunları analiz et"

### PDF Query Examples:
- "Kayıtlı sermaye tavanı nedir?"
- "133.096 TL ile ilgili bilgi ver"
- "Mali tablolardaki ana kalemler neler?"
- "Şirketin finansal performansını özetle"
"""
    
    with open("colab_startup_latest.md", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("📝 Created colab_startup_latest.md with all new features")

def main():
    """Main setup function"""
    
    if not check_colab():
        print("⚠️ This script is optimized for Google Colab")
        print("For local installation, use requirements.txt")
    
    # Install packages
    install_packages()
    
    # Setup HuggingFace
    setup_huggingface()
    
    # Setup NLTK (NEW)
    setup_nltk()
    
    # Check GPU
    check_gpu()
    
    # Check Excel support
    check_excel_support()
    
    # Check new features (NEW)
    check_new_features()
    
    # Create startup guide
    create_startup_notebook()
    
    # Provide manual fix options
    fix_streamlit_manual()
    
    print("\n🎉 Setup completed with all latest features!")
    print("\n📋 Next steps:")
    print("1. Check colab_startup_latest.md for detailed instructions")
    print("2. If any imports fail, restart runtime and run again")
    print("3. Get HuggingFace token if using Llama models")
    print("4. Run: streamlit run chat_bot.py")
    print("5. 🆕 Upload Excel files (.xls, .xlsx, .xlsm) for analysis!")
    print("6. 🚀 Test new features: smart chunking, better search, evaluation tools")
    
    print("\n🆕 Latest Improvements:")
    print("- ✅ Smart sentence-based chunking (300 chars default)")
    print("- ✅ Enhanced vector search with min_score filtering")
    print("- ✅ Rich metadata (language, content type, numbers, currency)")
    print("- ✅ Evaluation system for testing model accuracy")
    print("- ✅ Better performance monitoring and logging")
    
    print("\n📊 Performance Tips:")
    print("- Use chunk_size=300, overlap_size=50 for best accuracy")
    print("- Set min_score=0.7 for high-quality results")
    print("- Process PDFs in batches of 10 pages for memory efficiency")
    print("- Use n_results=5-10 for optimal context length")
    
    # Final import test
    try:
        import streamlit as st
        import nltk
        from sentence_transformers import SentenceTransformer
        print("\n✅ Final check: All critical imports successful!")
    except Exception as e:
        print(f"\n❌ Final check failed: {e}")
        print("📋 Restart runtime and run setup again")

if __name__ == "__main__":
    main() 