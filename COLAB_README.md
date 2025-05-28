# 🚀 Google Colab'da PDF ChatBot

Bu projeyi Google Colab'da çalıştırmak için step-by-step rehber.

## ⚡ Hızlı Başlangıç

### 1. Colab Notebook Oluştur
1. [Google Colab](https://colab.research.google.com) açın
2. New notebook oluşturun
3. GPU'yu aktifleştirin: `Runtime → Change runtime type → GPU → T4/V100/A100`

### 2. Projeyi İndirin
```python
!git clone https://github.com/YOUR_USERNAME/pdf-chatbot.git
%cd pdf-chatbot
```

### 3. Özel Colab Kurulumu
```python
# Dependency conflicts'i önleyen özel setup
!python colab_setup.py
```

### 4. Runtime'ı Restart Edin
- `Runtime → Restart runtime` 
- **Önemli:** Bu adımı atlarsanız import error'ları alabilirsiniz

### 5. ChatBot'u Başlatın
```python
!python chat_bot.py
```

### 6. Public URL Alın
Colab otomatik olarak public URL üretecek:
```
* Running on public URL: https://xxxxx.gradio.live
```

## 🔧 Dependency Conflicts Çözümü

Eğer halen conflict error alıyorsanız:

### Çözüm 1: Manuel Kurulum
```python
# Sadece gerekli paketleri yükle
!pip install --quiet transformers accelerate
!pip install --quiet sentence-transformers chromadb
!pip install --quiet pdfplumber PyPDF2 
!pip install --quiet gradio python-dotenv

# Runtime restart
# Sonra projeyi çalıştır
```

### Çözüm 2: Ignore Conflicts
```python
!pip install -r requirements_colab.txt --no-deps
```

### Çözüm 3: Fresh Runtime
```python
# Runtime → Factory reset runtime
# Sonra baştan başla
```

## 🤖 Model Kurulumu

### HuggingFace Token (Llama için)
```python
# 1. Token alın: https://huggingface.co/settings/tokens
# 2. Login yapın:
from huggingface_hub import login
login(token="hf_your_token_here")
```

### GPU Memory Kontrolü
```python
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Model Önerileri
- **T4 (16GB)**: Mistral 7B veya Llama 3.2 3B
- **V100 (32GB)**: Llama 3.1 8B
- **A100 (40GB)**: Llama 3.1 8B veya 70B

## 📱 Kullanım

1. **Model Seç**: Interface'de model dropdown'dan seçin
2. **ChatBot Başlat**: "🚀 ChatBot'u Başlat" butonuna tıklayın  
3. **PDF Yükle**: PDF dosyanızı yükleyin
4. **Soru Sor**: Chat'te sorularınızı yazın

## ⚠️ Yaygın Sorunlar

### "CUDA out of memory"
```python
# Çözüm: Daha küçük model veya quantization
# Model seçerken memory'nizi kontrol edin
```

### "Module not found"
```python
# Çözüm: Runtime restart
# Runtime → Restart runtime → Tekrar çalıştır
```

### "Gradio connection error"
```python
# Çözüm: Public URL kullanın
demo.launch(share=True)  # Otomatik olarak yapılıyor
```

### Model yükleme çok yavaş
```python
# İlk seferde normal (model indiriliyor)
# Sonraki seferlerde cache'den hızlı yüklenir
# Sabırla bekleyin: 2-5 dakika normal
```

## 🔒 Colab Limitations

- **Session Timeout**: 12 saat sonra kapanır
- **Disk Space**: ~100GB limit
- **GPU Time**: Ücretsiz hesapta limit var
- **Model Cache**: Session kapanınca silinir

## 💡 Pro Tips

### Model Cache Etme
```python
# Model'i persistent storage'a kaydet
from google.colab import drive
drive.mount('/content/drive')

# Cache directory'yi drive'a yönlendir
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```

### Background Running
```python
# ChatBot'u background'da çalıştır
!nohup python chat_bot.py > chatbot.log 2>&1 &

# Log'ları takip et
!tail -f chatbot.log
```

### Memory Monitoring
```python
# GPU memory kullanımını takip et
!watch -n 1 nvidia-smi  # Her saniye güncelle
```

## 🔄 Quick Commands

```python
# Hızlı restart ve çalıştırma
def quick_restart():
    import os
    os.kill(os.getpid(), 9)  # Runtime restart

# Memory temizleme
def clear_memory():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared!")

# ChatBot durumu
!ps aux | grep "chat_bot"
```

## 📞 Sorun mu var?

1. **Bu README'yi takip edin** step-by-step
2. **Runtime'ı restart edin** conflict'lerde  
3. **GPU memory'nizi kontrol edin** model seçerken
4. **Sabırlı olun** ilk model yüklemede

---

**🎉 Başarıyla çalışınca public URL'inizi paylaşabilirsiniz!** 