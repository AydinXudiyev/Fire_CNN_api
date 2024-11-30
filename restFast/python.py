from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import os

# FastAPI uygulaması
app = FastAPI()

# Model dosya yolu ve yüklenmesi
model_path = os.path.abspath("restFast/model_fire.keras")
if os.path.exists(model_path):
    print(f"Model bulundu: {model_path}")
else:
    raise RuntimeError(f"Model dosyası bulunamadı: {model_path}")

try:
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    raise RuntimeError(f"Model yüklenemedi: {e}")

# Sınıf etiketleri ve güvenlik önerileri
class_labels = ['Fire', 'Non-Fire']
fire_tips = {
    'Fire': "Yangın tespit edildi! Lütfen yetkililere haber verin.",
    'Non-Fire': "Güvende görünüyorsunuz. Ancak, şüphe varsa etrafı kontrol edin."
}

# Tahmin yapılacak görselin URL formatında alınması
class ImageURL(BaseModel):
    url: str

# Tahmin işlemi
@app.post("/predict")
def predict_fire(image_data: ImageURL):
    try:
        # Görseli URL'den indirme
        response = requests.get(image_data.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Görsel URL'ine ulaşılamadı.")
        
        try:
            img = Image.open(BytesIO(response.content))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Geçersiz görsel formatı.")

        # Görseli işleme (RGB'ye çevirme, boyutlandırma)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalizasyon
        img_array = np.expand_dims(img_array, axis=0)

        # Modelden tahmin alma
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = float(prediction[0][predicted_index])
        tips = fire_tips[predicted_class]

        # Tahmin sonucu döndürme
        return {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}",
            "tips": tips
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin işlemi sırasında hata oluştu: {e}")

# Ana sayfa
@app.get("/")
def read_root():
    return {"message": "Fire and Non-Fire sınıflandırma API'si çalışıyor!"} 

if _name_ == "_main_":
    port = int(os.getenv("PORT", 8000))  # Portu ortama göre al, yoksa 8000 olarak ayarla
    uvicorn.run(app, host="0.0.0.0", port=port)
