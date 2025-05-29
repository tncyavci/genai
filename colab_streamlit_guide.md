# 🚀 PDF ChatBot - Google Colab'da Streamlit ile Çalıştırma

## ⚡ Hızlı Başlangıç (2 dakikada çalışır!)

### 1. Colab Notebook Oluştur
```python
# Google Colab'da yeni notebook oluşturun
# Runtime → Change runtime type → Hardware accelerator → GPU (T4/V100/A100)
```

### 2. Projeyi Klonla ve Kurulum Yap
```python
# Projeyi indirin
!git clone YOUR_REPO_URL
%cd pdf-chatbot

# Özel Colab setup'ı çalıştırın (dependency conflicts'i çözer)
!python colab_setup.py

# Runtime'ı restart edin (önemli!)
# Runtime → Restart runtime
```

### 3. Streamlit ChatBot'u Başlatın
```python
# Streamlit uygulamasını başlatın
!streamlit run chat_bot.py --server.port 8501 &

# Alternatif: Debug mode ile
!streamlit run chat_bot.py --server.port 8501 --logger.level debug &
```

### 4. Public URL Alın
```python
# Streamlit otomatik olarak public URL oluşturur
# Terminal output'unda şöyle bir link göreceksiniz:
# External URL: https://xyz-abc123.streamlit.app

# Alternatif: ngrok ile custom domain
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"🌐 Public URL: {public_url}")
```

## 🔧 Model Kurulumu

### HuggingFace Token (Llama modeller için)
```python
# 1. Token alın: https://huggingface.co/settings/tokens
# 2. Login yapın:
from huggingface_hub import login
login(token="hf_your_token_here")
```

### GPU Memory Kontrolü
```python
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_memory:.1f}GB")
    
    # Model önerileri
    if gpu_memory > 20:
        print("🎯 Recommended: Llama 3.1 8B")
    elif gpu_memory > 15:
        print("🎯 Recommended: Mistral 7B")
    else:
        print("🎯 Recommended: Llama 3.2 3B (quantized)")
else:
    print("⚠️ GPU bulunamadı - CPU çok yavaş olacak")
```

## 📱 Kullanım Rehberi

### 1. Model Seçimi
- Streamlit arayüzünde sidebar'dan model seçin
- GPU memory'nize göre model seçin
- "🚀 ChatBot'u Başlat" butonuna tıklayın

### 2. PDF Yükleme
- "PDF Dosyası Yükleyin" bölümünden dosya seçin
- Dosya işlenmeyi bekleyin (1-2 dakika)
- ✅ işareti görünce hazır

### 3. Sohbet Etme
- "Sorunuzu yazın..." kutusuna sorunuzu yazın
- Enter'a basın veya Send butonuna tıklayın
- Model yanıtını bekleyin

## ⚠️ Yaygın Sorunlar ve Çözümler

### "Streamlit bulunamadı" hatası
```python
# Çözüm:
!pip install streamlit>=1.28.0
# Runtime → Restart runtime
```

### "CUDA out of memory" hatası
```python
# Çözüm 1: Memory temizle
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Çözüm 2: Daha küçük model seç
# Llama 3.2 3B veya quantized model kullan
```

### Streamlit bağlanamıyor
```python
# Port kontrolü
!netstat -tlnp | grep :8501

# Streamlit process'i öldür ve yeniden başlat
!pkill -f streamlit
!streamlit run chat_bot.py --server.port 8501 &
```

### Model yüklenmiyor
```python
# Internet bağlantısını kontrol edin
!ping huggingface.co

# Cache temizle
!rm -rf ~/.cache/huggingface/
!rm -rf /content/hf_cache/

# Token'ı yeniden girin
from huggingface_hub import login
login(token="your_token")
```

## 💡 Pro Tips

### 1. Model Cache'i Kaydet
```python
# Google Drive'a mount et
from google.colab import drive
drive.mount('/content/drive')

# Cache directory'yi drive'a yönlendir
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/hf_cache"
```

### 2. Background'da Çalıştır
```python
# Nohup ile background
!nohup streamlit run chat_bot.py --server.port 8501 > streamlit.log 2>&1 &

# Log'ları takip et
!tail -f streamlit.log
```

### 3. Memory Monitoring
```python
# GPU kullanımını izle
!watch -n 2 nvidia-smi

# Streamlit memory kullanımı
import psutil
import os
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### 4. Session State Koruma
```python
# Streamlit session state'i korumak için
# Uygulamayı yeniden başlatmadan model değiştirme
# Settings → Clear Cache → Restart
```

## 📊 Model Performans Karşılaştırması

| Model | Memory | Speed | Quality | Colab Uyumlu |
|-------|--------|-------|---------|---------------|
| Llama 3.1 8B | 16GB | Orta | Yüksek | ✅ V100/A100 |
| Mistral 7B | 14GB | Hızlı | İyi | ✅ T4/V100 |
| Llama 3.2 3B | 6GB | Hızlı | Orta | ✅ T4 |
| Quantized 4-bit | 4GB | Orta | Orta | ✅ T4 |

## 🔄 Hızlı Komutlar

```python
# Streamlit'i yeniden başlat
def restart_streamlit():
    !pkill -f streamlit
    !streamlit run chat_bot.py --server.port 8501 &

# Memory temizle
def clear_gpu_memory():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared!")

# Process kontrolü
def check_streamlit():
    !ps aux | grep streamlit
    !netstat -tlnp | grep :8501

# URL'i göster
def show_url():
    from pyngrok import ngrok
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        print(f"🌐 URL: {tunnel.public_url}")
```

## 🎯 Streamlit vs Gradio Avantajları

**Streamlit Avantajları:**
- ✅ Dependency conflict'leri yok
- ✅ Daha stabil session management
- ✅ Better file upload handling
- ✅ Professional UI/UX
- ✅ Real-time updates
- ✅ Better mobile support

**Gradio Sorunları (neden değiştirdik):**
- ❌ Dependency conflict'leri
- ❌ Session timeout sorunları
- ❌ File upload bugs
- ❌ Memory leak'ler

## 🚀 Başarılı Çalışma Kontrol Listesi

- [ ] ✅ GPU aktif (nvidia-smi)
- [ ] ✅ Dependencies kurulu (requirements)
- [ ] ✅ HuggingFace token set
- [ ] ✅ Streamlit çalışıyor (:8501)
- [ ] ✅ Public URL erişilebilir
- [ ] ✅ Model yüklendi
- [ ] ✅ PDF yükleme çalışıyor
- [ ] ✅ Chat respond ediyor

---

**🎉 Başarıyla çalıştırdığınızda arayüz şöyle görünecek:**
- Sidebar'da model seçenekleri
- PDF upload area
- Chat interface
- Real-time response
- Source citations

**Colab'da Streamlit = Mükemmel kombi! 🚀** 