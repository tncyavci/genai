# 🚀 Google Colab Pro Plus - Local LLM Kurulum Rehberi

Google Colab'da PDF ChatBot projenizi local LLM ile nasıl çalıştıracağınıza dair detaylı rehber.

## 🎯 Model Önerileri (Colab Pro Plus için)

### 1. **Llama 3.1 8B Instruct** (En İdeal)
```python
# HuggingFace Login gerekli (ücretsiz hesap)
!huggingface-cli login --token YOUR_HF_TOKEN

model_name = "meta-llama/Llama-3.1-8B-Instruct"
```

**Avantajları:**
- ✅ Türkçe desteği mükemmel
- ✅ Finansal analiz için optimize
- ✅ 16GB GPU memory (Colab Pro Plus uygun)
- ✅ Meta'nın en stabil modeli

### 2. **Mistral 7B Instruct** (Hızlı Alternatif)
```python
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
```

**Avantajları:**
- ✅ Daha az memory (~14GB)
- ✅ Hızlı inference
- ✅ Avrupa dilleri güçlü
- ⚠️ Türkçe Llama'dan biraz zayıf

### 3. **Llama 3.2 3B** (Hafif Seçenek)
```python
model_name = "meta-llama/Llama-3.2-3B-Instruct"
```

**Avantajları:**
- ✅ Çok hafif (~6GB)
- ✅ Hızlı başlatma
- ⚠️ Daha basit reasoning

## 🔧 Colab Kurulum Adımları

### 1. Gerekli Kütüphaneleri Yükleyin
```bash
!pip install transformers accelerate torch bitsandbytes
!pip install gradio chromadb sentence-transformers
!pip install pdfplumber PyPDF2 pandas numpy
```

### 2. GPU Kontrolü
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "No GPU")
```

### 3. HuggingFace Token (Llama için)
```python
# https://huggingface.co/settings/tokens adresinden token alın
from huggingface_hub import login
login(token="your_token_here")
```

### 4. Projeyi İndirin ve Çalıştırın
```python
!git clone YOUR_REPO_URL
%cd pdf-chatbot

# ChatBot'u başlatın
!python chat_bot.py
```

## 🎛️ Memory Optimizasyonları

### Quantization (4-bit/8-bit)
```python
# llm_service_local.py dosyasında
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,  # 8-bit quantization
    # load_in_4bit=True,  # Daha agresif
)
```

### Gradient Checkpointing
```python
self.model.gradient_checkpointing_enable()
```

### Memory Cleanup
```python
import gc
import torch

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
# Model değiştirirken kullanın
cleanup_memory()
```

## 🌐 Public URL (Colab)

```python
# chat_bot.py dosyasında
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Public URL için
    debug=False
)
```

## 📊 Performance Karşılaştırması

| Model | Memory | Hız | Türkçe | Reasoning | Colab Uyumlu |
|-------|--------|-----|--------|-----------|---------------|
| Llama 3.1 8B | 16GB | Orta | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| Mistral 7B | 14GB | Hızlı | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| Llama 3.2 3B | 6GB | Çok Hızlı | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| Llama 3.1 70B | 40GB | Yavaş | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |

## 🚫 Yaygın Hatalar ve Çözümleri

### 1. "CUDA out of memory"
```python
# Çözüm: Quantization kullanın
load_in_8bit=True
# veya daha küçük model seçin
```

### 2. "Gated repo" Hatası
```python
# Çözüm: HuggingFace login
!huggingface-cli login
# veya açık model kullanın
```

### 3. "Model too slow"
```python
# Çözüm: Smaller model veya quantization
model_name = "meta-llama/Llama-3.2-3B-Instruct"
```

## 🎯 Önerilen Colab Workflow

```python
# 1. Kütüphaneler
!pip install -r requirements.txt

# 2. GPU Check  
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 3. Model seçimi (memory'ye göre)
if torch.cuda.get_device_properties(0).total_memory > 20e9:
    model_choice = "llama_3_1_8b"  # 16GB+ için
else:
    model_choice = "llama_3_2_3b"  # Daha az için

# 4. ChatBot başlat
from chat_bot import create_gradio_interface
demo = create_gradio_interface()
demo.launch(share=True)  # Public URL
```

## 💡 Pro Tips

1. **Model Cache**: İlk yüklemede model indirilir, sonraki çalıştırmalarda cache'den alınır
2. **Session Persistence**: Colab session kapanırsa model yeniden yüklenir
3. **Background Running**: `nohup python chat_bot.py &` ile background'da çalıştırın
4. **Memory Monitor**: `!nvidia-smi` ile GPU memory takip edin

## 🔗 Faydalı Linkler

- [HuggingFace Token](https://huggingface.co/settings/tokens)
- [Llama Models](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- [Mistral Models](https://huggingface.co/collections/mistralai/mistral-7b-65f946c6b3aac8b1b8c11ce0)
- [Google Colab Pro](https://colab.research.google.com/signup)

---

Bu rehber ile Google Colab Pro Plus'ta local LLM ile PDF ChatBot'unuzu sorunsuz çalıştırabilirsiniz! 🚀 