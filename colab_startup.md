
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
