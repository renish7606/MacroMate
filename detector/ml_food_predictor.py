import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "food101_efficientnet.keras")
LABELS_PATH = os.path.join(BASE_DIR, "data", "food101_labels.txt")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 40
# ==========================================

# Load model once
model = load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r") as f:
    CLASS_LABELS = [line.strip() for line in f.readlines()]


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# def predict_food(image_path):
#     img = preprocess_image(image_path)
    
#     logits = model.predict(img)[0]
#     predictions = tf.nn.softmax(logits).numpy()

#     best_index = int(np.argmax(predictions))
#     confidence = float(predictions[best_index])

#     return {
#         "food_name": CLASS_LABELS[best_index],
#         "confidence": round(confidence, 2),
#         "is_confident": confidence >= CONFIDENCE_THRESHOLD,
#         "top_predictions": get_top_predictions(predictions)
#     }

def predict_food(image_path):
    img = preprocess_image(image_path)

    # Model output (logits)
    logits = model.predict(img)[0]

    # Convert logits → probabilities
    # predictions = tf.nn.softmax(logits).numpy()
    predictions = softmax_with_temperature(logits, temperature=1.5)

    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])

    return {
        "food_name": CLASS_LABELS[best_index],
        "confidence": round(confidence * 100, 2),  # %
        "is_confident": confidence >= 0.40,
        "top_predictions": [
            {
                "food": CLASS_LABELS[i],
                "confidence": round(float(predictions[i]) * 100, 2)
            }
            for i in np.argsort(predictions)[-5:][::-1]
        ]
    }


def get_top_predictions(predictions, top_k=5):
    indices = np.argsort(predictions)[-top_k:][::-1]
    return [
        {
            "food": CLASS_LABELS[i],
            "confidence": round(float(predictions[i])*100, 2)
        }
        for i in indices
        
    ]

def softmax_with_temperature(logits, temperature=1.5):
    exp = np.exp(logits / temperature)
    return exp / np.sum(exp)