"""
MacroMate Hybrid Food Detector
───────────────────────────────
Priority 1: Custom trained model  (final_model.h5)
Priority 2: CLIP pre-trained model (fallback)

Terminal output shows which model was used for every prediction.
"""

import numpy as np
import json
import os
from PIL import Image

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.h5")
CLASS_IDX_PATH    = os.path.join(BASE_DIR, "models", "class_indices.json")
NUTRITION_PATH    = os.path.join(BASE_DIR, "models", "nutrition_lookup.json")

# ─────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────
CUSTOM_CONFIDENCE_THRESHOLD = 0.55   # custom model wins above this
CLIP_CONFIDENCE_THRESHOLD   = 0.30   # CLIP result accepted above this
IMG_SIZE                    = 224

# ─────────────────────────────────────────
# LOAD CUSTOM MODEL + MAPPINGS
# ─────────────────────────────────────────
custom_model   = None
INDEX_TO_FOOD  = {}   # {class_index: "food_name"}
NUTRITION_DB   = {}   # {"food_name": {calories, protein, ...}}

def _load_custom_model():
    """Load the trained Keras model and class mappings. Called once at import."""
    global custom_model, INDEX_TO_FOOD, NUTRITION_DB

    # ── class_indices.json ──
    if os.path.exists(CLASS_IDX_PATH):
        with open(CLASS_IDX_PATH, "r", encoding="utf-8") as f:
            food_to_idx = json.load(f)
        # Build reverse map: index → food name
        INDEX_TO_FOOD = {v: k for k, v in food_to_idx.items()}
        print(f"[MODEL LOADER] class_indices loaded: {len(INDEX_TO_FOOD)} classes")
    else:
        print(f"[MODEL LOADER] WARNING: class_indices.json not found at {CLASS_IDX_PATH}")

    # ── nutrition_lookup.json ──
    if os.path.exists(NUTRITION_PATH):
        with open(NUTRITION_PATH, "r", encoding="utf-8") as f:
            NUTRITION_DB = json.load(f)
        print(f"[MODEL LOADER] nutrition_lookup loaded: {len(NUTRITION_DB)} foods")
    else:
        print(f"[MODEL LOADER] WARNING: nutrition_lookup.json not found at {NUTRITION_PATH}")

    # ── final_model.h5 ──
    if not os.path.exists(CUSTOM_MODEL_PATH):
        print(f"[MODEL LOADER] WARNING: final_model.h5 not found at {CUSTOM_MODEL_PATH}")
        print(f"[MODEL LOADER] Falling back to built-in vision detector.")
        return

    try:
        import tensorflow as tf
        custom_model = tf.keras.models.load_model(CUSTOM_MODEL_PATH)
        print(f"[MODEL LOADER] Custom vision model loaded: {CUSTOM_MODEL_PATH}")
        print(f"[MODEL LOADER]   Input shape:  {custom_model.input_shape}")
        print(f"[MODEL LOADER]   Output shape: {custom_model.output_shape}")
    except Exception as e:
        print(f"[MODEL LOADER] Custom vision model failed to load: {e}")
        print(f"[MODEL LOADER] Falling back to built-in vision detector.")
        custom_model = None


# ─────────────────────────────────────────
# LOAD CLIP (unchanged from original)
# ─────────────────────────────────────────
CLIP_AVAILABLE = False
clip_model     = None
clip_preprocess = None
device         = "cpu"

try:
    import torch
    import clip as openai_clip
    device          = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)
    CLIP_AVAILABLE  = True
    print(f"[MODEL LOADER] Vision detector loaded on {device}")
except ImportError:
    print("[MODEL LOADER] CLIP not installed -- install with: "
          "pip install git+https://github.com/openai/CLIP.git")
except Exception as e:
    print(f"[MODEL LOADER] CLIP failed to load: {e}")


# ─────────────────────────────────────────
# CLIP FOOD LIST (keep existing comprehensive list)
# ─────────────────────────────────────────
CLIP_FOOD_LIST = [
    # Indian Foods
    "samosa", "dosa", "idli", "vada", "biryani", "pakoda", "paneer tikka",
    "butter chicken", "palak paneer", "tandoori chicken", "naan bread",
    "roti", "paratha", "chapati", "puri", "bhaji", "pav bhaji",
    "chole", "dal", "rajma", "chicken curry", "fish curry", "korma",
    "dhokla", "jalebi", "kachori", "panipuri", "cholebhature", "dabeli",
    "kathiroll", "kofta", "naan", "paneer", "pavbhaji", "vadapav",
    # Chinese
    "dim sum", "baozi", "dumplings", "fried rice", "lo mein", "chow mein",
    "spring rolls", "egg rolls", "wonton soup", "hot and sour soup",
    "kung pao chicken", "sweet and sour pork", "peking duck",
    # Mexican
    "tacos", "burritos", "enchiladas", "quesadilla", "nachos", "guacamole",
    # Italian
    "pizza", "pasta", "lasagna", "risotto", "spaghetti", "ravioli",
    "gnocchi", "cannoli", "tiramisu", "panna cotta", "bruschetta",
    # American
    "hamburger", "hot dog", "french fries", "fried chicken", "mac and cheese",
    "grilled cheese sandwich", "pancakes", "waffles", "donuts", "apple pie",
    # Japanese
    "sushi", "sashimi", "ramen", "udon", "tempura", "teriyaki",
    "takoyaki", "miso soup",
    # Middle Eastern
    "falafel", "hummus", "shawarma", "kebab",
    # Thai
    "pad thai", "green curry",
    # Breakfast
    "eggs benedict", "omelette", "french toast", "breakfast burrito",
    # Desserts
    "ice cream", "cake", "cookies", "cheesecake", "chocolate mousse",
    "macarons", "creme brulee",
    # Soups / Salads
    "caesar salad", "greek salad", "clam chowder", "pho",
    "steak", "grilled salmon", "scallops", "oysters", "mussels",
    "chicken wings", "beef tartare", "foie gras", "escargots",
]

# Run loaders at import time
_load_custom_model()


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def _preprocess_for_custom(image_path: str):
    """
    Preprocess image for the custom Keras model.
    Uses EfficientNet-style preprocessing (scale to [-1, 1]).
    """
    import tensorflow as tf
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────
# CUSTOM MODEL PREDICTOR
# ─────────────────────────────────────────
def _predict_custom(image_path: str) -> dict | None:
    """
    Run prediction with the custom trained model.
    Returns result dict or None if not confident / error.
    """
    if custom_model is None:
        return None
    if not INDEX_TO_FOOD:
        print("[CUSTOM MODEL] No class index map -- skipping custom model")
        return None

    try:
        img  = _preprocess_for_custom(image_path)
        preds = custom_model.predict(img, verbose=0)[0]

        # softmax if output looks like logits
        if not (0.98 < preds.sum() < 1.02):
            import tensorflow as tf
            preds = tf.nn.softmax(preds).numpy()

        best_idx    = int(np.argmax(preds))
        confidence  = float(preds[best_idx])
        food_name   = INDEX_TO_FOOD.get(best_idx, None)

        if food_name is None:
            print(f"[CUSTOM MODEL] Index {best_idx} not in class map -- skipping")
            return None

        # Top-5 predictions
        top5_idx = np.argsort(preds)[-5:][::-1]
        top5 = [
            {
                "food":       INDEX_TO_FOOD.get(int(i), f"class_{i}"),
                "confidence": round(float(preds[i]) * 100, 2),
            }
            for i in top5_idx if int(i) in INDEX_TO_FOOD
        ]

        return {
            "food_name":       food_name.lower().replace(" ", "_"),
            "confidence":      round(confidence * 100, 2),
            "is_confident":    confidence >= CUSTOM_CONFIDENCE_THRESHOLD,
            "top_predictions": top5,
            "method":          "CUSTOM",
        }

    except Exception as e:
        print(f"[CUSTOM MODEL] Prediction error: {e}")
        return None


# ─────────────────────────────────────────
# CLIP PREDICTOR (unchanged logic)
# ─────────────────────────────────────────
def _predict_clip(image_path: str) -> dict | None:
    """
    Run prediction with CLIP.
    Returns result dict or None if CLIP unavailable.
    """
    if not CLIP_AVAILABLE or clip_model is None:
        return None

    try:
        import torch
        import clip as openai_clip

        image       = Image.open(image_path).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        text_prompts = [f"a photo of {food}" for food in CLIP_FOOD_LIST]
        text_inputs  = openai_clip.tokenize(text_prompts).to(device)

        with torch.no_grad():
            img_feat  = clip_model.encode_image(image_input)
            txt_feat  = clip_model.encode_text(text_inputs)
            similarity = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            values, indices = similarity[0].topk(10)

        results = []
        for value, idx in zip(values, indices):
            food       = CLIP_FOOD_LIST[idx]
            conf       = float(value) * 100
            results.append({"food": food.title(), "confidence": conf})

        best = results[0]
        return {
            "food_name":       best["food"].lower().replace(" ", "_"),
            "confidence":      round(best["confidence"], 2),
            "is_confident":    best["confidence"] >= CLIP_CONFIDENCE_THRESHOLD * 100,
            "top_predictions": results[:5],
            "method":          "CLIP",
        }

    except Exception as e:
        print(f"[CLIP] Prediction error: {e}")
        return None


# ─────────────────────────────────────────
# PUBLIC API — called by views.py
# ─────────────────────────────────────────
def predict_food(image_path: str) -> dict:
    """
    Hybrid predictor. Tries custom model first, falls back to CLIP.
    Always logs which model was used to terminal.
    """
    print(f"\n{'─'*60}")
    print(f"[PREDICTOR] Image received: {os.path.basename(image_path)}")

    # ── 1. Try custom model ──
    if custom_model is not None:
        custom_result = _predict_custom(image_path)

        if custom_result and custom_result["is_confident"]:
            food = custom_result["food_name"].replace("_", " ")
            conf = custom_result["confidence"]
            print(f"[DETECTOR] High-confidence prediction: {food.title()} ({conf}%)")
            print(f"{'─'*60}\n")
            return custom_result

        elif custom_result:
            food = custom_result["food_name"].replace("_", " ")
            conf = custom_result["confidence"]
            print(f"[DETECTOR] Low confidence from primary model: {food.title()} ({conf}%) "
                  f"-- using backup detector")
    else:
        print("[DETECTOR] Primary model not available -- using backup detector directly")

    # ── 2. Fall back to CLIP ──
    if CLIP_AVAILABLE:
        clip_result = _predict_clip(image_path)
        if clip_result:
            food = clip_result["food_name"].replace("_", " ")
            conf = clip_result["confidence"]
            reason = ("primary model unavailable"
                      if custom_model is None
                      else "primary model low confidence")
            print(f"[DETECTOR] Backup prediction: {food.title()} ({conf}%) -- {reason}")
            print(f"{'─'*60}\n")
            return clip_result

    # ── 3. Both failed ──
    print("[PREDICTOR] Both models failed -- returning unknown")
    print(f"{'─'*60}\n")
    return {
        "food_name":       "unknown",
        "confidence":      0.0,
        "is_confident":    False,
        "top_predictions": [],
        "method":          "NONE",
        "error":           "All models failed",
    }


# ─────────────────────────────────────────
# NUTRITION LOOKUP (new — uses nutrition_lookup.json)
# ─────────────────────────────────────────
def get_nutrition_from_model_db(food_name: str) -> dict | None:
    """
    Look up nutrition from nutrition_lookup.json (richer than calories.csv).
    Returns dict with calories, protein, carbs, fat, fiber, sugar,
    vitamin_a, vitamin_c, calcium, iron — or None if not found.
    """
    if not NUTRITION_DB:
        return None

    key = food_name.lower().replace("_", " ")

    # Direct match
    if key in NUTRITION_DB:
        entry = NUTRITION_DB[key]
        return {
            "calories": int(entry.get("calories", 0)),
            "protein":  round(float(entry.get("protein",  0)), 1),
            "carbs":    round(float(entry.get("carbohydrates", 0)), 1),
            "fat":      round(float(entry.get("fat",      0)), 1),
            "fiber":    round(float(entry.get("fiber",    0)), 1),
            "sugar":    round(float(entry.get("sugar",    0)), 1),
            "vitamin_a": round(float(entry.get("vitamin_a", 0)), 1),
            "vitamin_c": round(float(entry.get("vitamin_c", 0)), 1),
            "calcium":   round(float(entry.get("calcium",   0)), 1),
            "iron":      round(float(entry.get("iron",      0)), 1),
        }

    # Partial match (e.g. "chicken_curry" → "chicken curry")
    for db_key in NUTRITION_DB:
        if key in db_key or db_key in key:
            entry = NUTRITION_DB[db_key]
            return {
                "calories": int(entry.get("calories", 0)),
                "protein":  round(float(entry.get("protein",  0)), 1),
                "carbs":    round(float(entry.get("carbohydrates", 0)), 1),
                "fat":      round(float(entry.get("fat",      0)), 1),
                "fiber":    round(float(entry.get("fiber",    0)), 1),
                "sugar":    round(float(entry.get("sugar",    0)), 1),
                "vitamin_a": round(float(entry.get("vitamin_a", 0)), 1),
                "vitamin_c": round(float(entry.get("vitamin_c", 0)), 1),
                "calcium":   round(float(entry.get("calcium",   0)), 1),
                "iron":      round(float(entry.get("iron",      0)), 1),
            }

    return None


# Legacy compatibility — old code calls softmax_with_temperature
def softmax_with_temperature(logits, temperature=1.0):
    exp = np.exp(logits / temperature)
    return exp / np.sum(exp)
