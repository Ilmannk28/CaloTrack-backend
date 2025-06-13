# backend/services/predictor.py
from ultralytics import YOLO
import cv2
from pathlib import Path

calories_dict = {
    'white rice': 200,
    'fried chicken': 250,
    'boiled egg': 70,
    'milk': 150,
    'sliced watermelon': 50
}

# Buat path absolut ke model
base_path = Path(__file__).resolve().parent
model_path = base_path.parent / "model" / "best.pt"

if not model_path.exists():
    raise FileNotFoundError(f"Model '{model_path}' tidak ditemukan.")

# Load model hanya sekali
model = YOLO(str(model_path))

def predict_calories(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar dari path: {image_path}")

    results = model.predict(image_path, imgsz=640, conf=0.25)
    predictions = []
    total_calories = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            calories = calories_dict.get(class_name, 0)

            # Ambil bounding box [x, y, w, h]
            bbox = box.xywh[0].tolist()
            x_center, y_center, width, height = bbox
            x = x_center - (width / 2)
            y = y_center - (height / 2)

            predictions.append({
                "food": class_name,
                "calories": calories,
                "confidence": round(confidence, 2),
                "bbox": [x, y, width, height]
            })
            total_calories += calories

    return {
        "predictions": predictions,
        "total_calories": total_calories
    }

