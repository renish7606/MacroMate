---
name: macromate-pytorch-model
description: >
  MacroMate new PyTorch model integration skill. Use this when replacing the old
  final_model.h5 (Keras/TensorFlow) with the new best_model.pt (PyTorch EfficientNet)
  trained on Food-101 + Indian foods + UECFood256. Covers: placing the 3 new files,
  completely rewriting ml_food_predictor.py to load best_model.pt with torchvision,
  hybrid detection logic (custom model first → CLIP fallback), nutrition lookup from
  label_nutrition_mapping.json, showing model name + confidence badge to the user in
  result.html and multi_result.html, and updating nutrition_provider.py.
  Apply alongside macromate-ui skill for visual consistency.
---

# MacroMate PyTorch Model Integration Skill

## What Changed

| Old | New |
|---|---|
| `final_model.h5` — Keras/TF, loading errors | `best_model.pt` — PyTorch, ~214 MB |
| `class_indices.json` — class → index map | `class_names.json` — ordered class list |
| `nutrition_lookup.json` — basic nutrition | `label_nutrition_mapping.json` — 103 KB richer data |

---

## Step 1 — Place the 3 New Files

Copy all 3 files into the `models/` folder (same place as the old ones):

```
FoodCalorieApp/models/
├── best_model.pt               ← NEW (PyTorch, ~214 MB)
├── class_names.json            ← NEW (ordered list of class names)
├── label_nutrition_mapping.json ← NEW (nutrition per label, 103 KB)
│
│ ── Keep old files for now (they won't be loaded anymore) ──
├── final_model.h5              ← old, ignored
├── class_indices.json          ← old, ignored
└── nutrition_lookup.json       ← old, ignored
```

Add to `.gitignore`:
```
models/*.pt
models/*.h5
```

---

## Step 2 — Install PyTorch + torchvision

Run in your virtualenv:

```bash
# CPU-only (if no GPU) — recommended for development
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
```

If you already have PyTorch installed (CLIP uses it), just add torchvision:
```bash
pip install torchvision
```

---

## Step 3 — Completely Rewrite `detector/ml_food_predictor.py`

Replace the **entire file** with the following:

```python
"""
MacroMate Hybrid Food Detector v2
──────────────────────────────────
Priority 1: Custom trained PyTorch model (best_model.pt)
            - EfficientNet trained on Food-101 + Indian foods + UECFood256
            - Used when predicted class is in class_names.json
Priority 2: CLIP pre-trained fallback
            - Used when custom model confidence < threshold
            - Used when custom model is unavailable

Terminal logs show which model was used, confidence %, and detected food.
User sees a badge in the UI: "Custom Model" or "CLIP Fallback".
"""

import os
import json
import numpy as np
from PIL import Image

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTORCH_MODEL_PATH    = os.path.join(BASE_DIR, "models", "best_model.pt")
CLASS_NAMES_PATH      = os.path.join(BASE_DIR, "models", "class_names.json")
NUTRITION_MAP_PATH    = os.path.join(BASE_DIR, "models", "label_nutrition_mapping.json")

# ─────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────
CUSTOM_CONFIDENCE_THRESHOLD = 0.50   # custom model wins if confidence >= 50%
CLIP_CONFIDENCE_THRESHOLD   = 0.25   # CLIP result shown if >= 25%
IMG_SIZE                    = 224

# ─────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────
custom_model   = None      # PyTorch model
CLASS_NAMES    = []        # ["apple_pie", "baby_back_ribs", ...]
NUTRITION_DB   = {}        # {"apple_pie": {calories, protein, ...}}


# ─────────────────────────────────────────
# LOADER — called ONCE at Django startup
# ─────────────────────────────────────────
def _load_custom_model():
    global custom_model, CLASS_NAMES, NUTRITION_DB

    # ── 1. class_names.json ──
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both list format ["apple_pie", ...] and dict format {"0": "apple_pie", ...}
        if isinstance(data, list):
            CLASS_NAMES = data
        elif isinstance(data, dict):
            # Sort by key to get ordered list
            CLASS_NAMES = [data[k] for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)]
        print(f"[MODEL LOADER] class_names.json loaded: {len(CLASS_NAMES)} classes")
        print(f"[MODEL LOADER]   Sample classes: {CLASS_NAMES[:5]}")
    else:
        print(f"[MODEL LOADER] MISSING: {CLASS_NAMES_PATH}")

    # ── 2. label_nutrition_mapping.json ──
    if os.path.exists(NUTRITION_MAP_PATH):
        with open(NUTRITION_MAP_PATH, "r", encoding="utf-8") as f:
            NUTRITION_DB = json.load(f)
        print(f"[MODEL LOADER] label_nutrition_mapping.json loaded: {len(NUTRITION_DB)} foods")
    else:
        print(f"[MODEL LOADER] MISSING: {NUTRITION_MAP_PATH}")

    # ── 3. best_model.pt ──
    if not os.path.exists(PYTORCH_MODEL_PATH):
        size_mb = 0
        print(f"[MODEL LOADER] MISSING: {PYTORCH_MODEL_PATH}")
        print(f"[MODEL LOADER] Custom model unavailable — CLIP will handle all predictions")
        return

    size_mb = os.path.getsize(PYTORCH_MODEL_PATH) / (1024 * 1024)
    print(f"[MODEL LOADER] Found best_model.pt: {size_mb:.1f} MB")

    try:
        import torch
        import torchvision.models as models

        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = len(CLASS_NAMES) if CLASS_NAMES else 372

        print(f"[MODEL LOADER] Loading PyTorch model on {device}...")
        print(f"[MODEL LOADER] Expected output classes: {num_classes}")

        # ── Try loading as full saved model first ──
        try:
            custom_model = torch.load(
                PYTORCH_MODEL_PATH,
                map_location=device,
                weights_only=False
            )
            custom_model.eval()
            print(f"[MODEL LOADER] Loaded as full saved model")

        except Exception as e1:
            print(f"[MODEL LOADER] Full model load failed ({e1}), trying state_dict...")

            # ── Try loading as state dict (weights only) ──
            try:
                # Build EfficientNet-B0 architecture (most common for food classification)
                model = models.efficientnet_b0(weights=None)
                # Replace classifier head to match training classes
                import torch.nn as nn
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

                state = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True)
                # Handle common state_dict wrapper keys
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]

                model.load_state_dict(state, strict=False)
                model = model.to(device)
                model.eval()
                custom_model = model
                print(f"[MODEL LOADER] Loaded as EfficientNet-B0 state_dict")

            except Exception as e2:
                print(f"[MODEL LOADER] state_dict load failed ({e2}), trying EfficientNet-B3...")

                # ── Try EfficientNet-B3 (larger, better accuracy) ──
                try:
                    import torch.nn as nn
                    model = models.efficientnet_b3(weights=None)
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                    state = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True)
                    if isinstance(state, dict):
                        for k in ["model_state_dict", "state_dict", "model"]:
                            if k in state:
                                state = state[k]
                                break
                    model.load_state_dict(state, strict=False)
                    model = model.to(device)
                    model.eval()
                    custom_model = model
                    print(f"[MODEL LOADER] Loaded as EfficientNet-B3 state_dict")

                except Exception as e3:
                    print(f"[MODEL LOADER] All load strategies failed.")
                    print(f"[MODEL LOADER]   Strategy 1 (full model): {e1}")
                    print(f"[MODEL LOADER]   Strategy 2 (B0 state_dict): {e2}")
                    print(f"[MODEL LOADER]   Strategy 3 (B3 state_dict): {e3}")
                    print(f"[MODEL LOADER] CLIP will be used as sole detector.")
                    custom_model = None
                    return

        print(f"[MODEL LOADER] ✓ Custom PyTorch model ready")

    except ImportError:
        print(f"[MODEL LOADER] PyTorch not installed. Run: pip install torch torchvision")
        custom_model = None
    except Exception as e:
        print(f"[MODEL LOADER] Unexpected error: {e}")
        custom_model = None


# ─────────────────────────────────────────
# LOAD CLIP
# ─────────────────────────────────────────
CLIP_AVAILABLE  = False
clip_model      = None
clip_preprocess = None
clip_device     = "cpu"

try:
    import torch
    import clip as openai_clip
    clip_device     = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=clip_device)
    CLIP_AVAILABLE  = True
    print(f"[MODEL LOADER] ✓ CLIP loaded on {clip_device}")
except ImportError:
    print("[MODEL LOADER] CLIP not installed — pip install git+https://github.com/openai/CLIP.git")
except Exception as e:
    print(f"[MODEL LOADER] CLIP failed: {e}")

# ─────────────────────────────────────────
# CLIP FOOD LIST (comprehensive)
# ─────────────────────────────────────────
CLIP_FOOD_LIST = [
    # Indian
    "samosa", "dosa", "idli", "vada", "biryani", "pakoda", "paneer tikka",
    "butter chicken", "palak paneer", "tandoori chicken", "naan bread",
    "roti", "paratha", "chapati", "puri", "bhaji", "pav bhaji",
    "chole", "dal", "rajma", "chicken curry", "fish curry", "korma",
    "dhokla", "jalebi", "kachori", "panipuri", "cholebhature", "dabeli",
    "kathiroll", "kofta", "naan", "paneer", "pavbhaji", "vadapav",
    # Chinese
    "dim sum", "dumplings", "fried rice", "chow mein", "spring rolls",
    "wonton soup", "hot and sour soup", "kung pao chicken", "peking duck",
    # Mexican
    "tacos", "burritos", "quesadilla", "nachos", "guacamole",
    # Italian
    "pizza", "pasta", "lasagna", "risotto", "spaghetti", "ravioli",
    "gnocchi", "cannoli", "tiramisu", "panna cotta", "bruschetta",
    # American
    "hamburger", "hot dog", "french fries", "fried chicken", "mac and cheese",
    "grilled cheese sandwich", "pancakes", "waffles", "donuts", "apple pie",
    # Japanese
    "sushi", "sashimi", "ramen", "udon", "tempura", "takoyaki", "miso soup",
    # Middle Eastern
    "falafel", "hummus", "shawarma", "kebab",
    # Thai
    "pad thai", "green curry",
    # Breakfast
    "eggs benedict", "omelette", "french toast", "breakfast burrito",
    # Desserts
    "ice cream", "cheesecake", "chocolate mousse", "macarons", "creme brulee",
    # Soups / Salads
    "caesar salad", "greek salad", "clam chowder", "pho",
    "steak", "grilled salmon", "scallops", "oysters", "mussels",
    "chicken wings", "beef tartare", "foie gras", "escargots",
]

# ─────────────────────────────────────────
# Run loaders at import time (once at startup)
# ─────────────────────────────────────────
_load_custom_model()


# ─────────────────────────────────────────
# PREPROCESSING FOR PYTORCH MODEL
# ─────────────────────────────────────────
def _preprocess_pytorch(image_path: str):
    """
    Preprocess image for the PyTorch EfficientNet model.
    Uses ImageNet normalization (standard for torchvision models).
    """
    import torch
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225]     # ImageNet std
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # add batch dimension

    device = next(custom_model.parameters()).device
    return tensor.to(device)


# ─────────────────────────────────────────
# CUSTOM MODEL PREDICTION
# ─────────────────────────────────────────
def _predict_custom(image_path: str) -> dict | None:
    """
    Run inference with the custom PyTorch model.
    Returns result dict or None on failure.
    """
    if custom_model is None or not CLASS_NAMES:
        return None

    try:
        import torch

        tensor = _preprocess_pytorch(image_path)

        with torch.no_grad():
            outputs = custom_model(tensor)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        probs_np = probs.cpu().numpy()
        best_idx = int(np.argmax(probs_np))

        if best_idx >= len(CLASS_NAMES):
            print(f"[CUSTOM MODEL] Index {best_idx} out of range ({len(CLASS_NAMES)} classes)")
            return None

        food_name  = CLASS_NAMES[best_idx]
        confidence = float(probs_np[best_idx])

        # Top-5 predictions
        top5_idx = np.argsort(probs_np)[-5:][::-1]
        top5 = [
            {
                "food":       CLASS_NAMES[i].replace("_", " ").title(),
                "confidence": round(float(probs_np[i]) * 100, 2),
            }
            for i in top5_idx if i < len(CLASS_NAMES)
        ]

        return {
            "food_name":       food_name.lower().replace(" ", "_"),
            "display_name":    food_name.replace("_", " ").title(),
            "confidence":      round(confidence * 100, 2),
            "is_confident":    confidence >= CUSTOM_CONFIDENCE_THRESHOLD,
            "top_predictions": top5,
            "method":          "CUSTOM",
            "method_label":    "Custom Model",
        }

    except Exception as e:
        print(f"[CUSTOM MODEL] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────
# CLIP PREDICTION
# ─────────────────────────────────────────
def _predict_clip(image_path: str) -> dict | None:
    """
    Run CLIP zero-shot classification.
    Returns result dict or None if CLIP unavailable.
    """
    if not CLIP_AVAILABLE or clip_model is None:
        return None

    try:
        import torch
        import clip as openai_clip

        image       = Image.open(image_path).convert("RGB")
        img_input   = clip_preprocess(image).unsqueeze(0).to(clip_device)
        text_prompts = [f"a photo of {food}" for food in CLIP_FOOD_LIST]
        text_inputs  = openai_clip.tokenize(text_prompts).to(clip_device)

        with torch.no_grad():
            img_feat  = clip_model.encode_image(img_input)
            txt_feat  = clip_model.encode_text(text_inputs)
            sim       = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            vals, idx = sim[0].topk(10)

        results = [
            {
                "food":       CLIP_FOOD_LIST[i].title(),
                "confidence": round(float(v) * 100, 2),
            }
            for v, i in zip(vals, idx)
        ]

        best = results[0]
        return {
            "food_name":       best["food"].lower().replace(" ", "_"),
            "display_name":    best["food"].title(),
            "confidence":      best["confidence"],
            "is_confident":    best["confidence"] >= CLIP_CONFIDENCE_THRESHOLD * 100,
            "top_predictions": results[:5],
            "method":          "CLIP",
            "method_label":    "CLIP Fallback",
        }

    except Exception as e:
        print(f"[CLIP] Prediction error: {e}")
        return None


# ─────────────────────────────────────────
# PUBLIC API — called by views.py
# ─────────────────────────────────────────
def predict_food(image_path: str) -> dict:
    """
    Hybrid predictor.
    1. Try custom PyTorch model → if confidence >= 50%, use it
    2. Otherwise fall back to CLIP
    Always logs which model was used to terminal.
    """
    print(f"\n{'─'*60}")
    print(f"[PREDICTOR] Image: {os.path.basename(image_path)}")

    # ── Try custom model ──
    if custom_model is not None:
        result = _predict_custom(image_path)

        if result and result["is_confident"]:
            print(f"[CUSTOM MODEL] ✓ {result['display_name']} ({result['confidence']}%)")
            print(f"{'─'*60}\n")
            return result

        elif result:
            print(f"[CUSTOM MODEL] Low confidence: {result['display_name']} "
                  f"({result['confidence']}%) — trying CLIP")
        else:
            print(f"[CUSTOM MODEL] Prediction failed — trying CLIP")
    else:
        print(f"[CUSTOM MODEL] Not loaded — using CLIP directly")

    # ── Fall back to CLIP ──
    if CLIP_AVAILABLE:
        clip_result = _predict_clip(image_path)
        if clip_result:
            reason = "custom model unavailable" if custom_model is None else "custom model low confidence"
            print(f"[CLIP FALLBACK] ✓ {clip_result['display_name']} ({clip_result['confidence']}%) — {reason}")
            print(f"{'─'*60}\n")
            return clip_result

    # ── Both failed ──
    print(f"[PREDICTOR] ✗ All models failed")
    print(f"{'─'*60}\n")
    return {
        "food_name":       "unknown",
        "display_name":    "Unknown",
        "confidence":      0.0,
        "is_confident":    False,
        "top_predictions": [],
        "method":          "NONE",
        "method_label":    "No Model",
        "error":           "All models failed",
    }


# ─────────────────────────────────────────
# NUTRITION LOOKUP — from label_nutrition_mapping.json
# ─────────────────────────────────────────
def get_nutrition_from_model_db(food_name: str) -> dict | None:
    """
    Look up nutrition from label_nutrition_mapping.json.
    Returns dict with: calories, protein, carbs, fat, fiber, sugar,
    vitamin_a, vitamin_c, calcium, iron — or None if not found.
    Tries exact match, then underscore/space variants, then partial match.
    """
    if not NUTRITION_DB:
        return None

    # Normalize key
    key = food_name.lower().strip()
    key_underscored = key.replace(" ", "_")
    key_spaced      = key.replace("_", " ")

    def extract(entry: dict) -> dict:
        """Normalize field names — handles both snake_case and CamelCase keys."""
        def g(names, default=0.0):
            for n in names:
                if n in entry:
                    try:
                        return round(float(entry[n]), 1)
                    except (TypeError, ValueError):
                        pass
            return default

        calories = g(["calories", "Calories", "kcal", "energy"])
        protein  = g(["protein", "Protein"])
        carbs    = g(["carbohydrates", "carbs", "Carbohydrates", "Carbs"])
        fat      = g(["fat", "Fat", "total_fat"])
        fiber    = g(["fiber", "Fiber", "dietary_fiber"])
        sugar    = g(["sugar", "Sugar", "sugars"])
        vit_a    = g(["vitamin_a", "Vitamin_A", "vitaminA"])
        vit_c    = g(["vitamin_c", "Vitamin_C", "vitaminC"])
        calcium  = g(["calcium", "Calcium"])
        iron     = g(["iron", "Iron"])

        return {
            "calories": int(calories),
            "protein":  protein,
            "carbs":    carbs,
            "fat":      fat,
            "fiber":    fiber,
            "sugar":    sugar,
            "vitamin_a": vit_a,
            "vitamin_c": vit_c,
            "calcium":   calcium,
            "iron":      iron,
        }

    # 1. Exact match
    for k in [key, key_underscored, key_spaced]:
        if k in NUTRITION_DB:
            return extract(NUTRITION_DB[k])

    # 2. Partial match (substring)
    for db_key in NUTRITION_DB:
        db_clean = db_key.lower().replace("_", " ")
        food_clean = key_spaced
        if food_clean in db_clean or db_clean in food_clean:
            return extract(NUTRITION_DB[db_key])

    return None


# Legacy compatibility
def softmax_with_temperature(logits, temperature=1.0):
    exp = np.exp(logits / temperature)
    return exp / np.sum(exp)
```

---

## Step 4 — Update `services/nutrition_provider.py`

Replace the file:

```python
"""
Nutrition Provider — v2
Priority 1: label_nutrition_mapping.json  (new, richest data)
Priority 2: Spoonacular API               (live, if USE_API_NUTRITION = True)
Priority 3: calories.csv                  (offline fallback)
"""
from django.conf import settings


def get_nutrition(food_name: str) -> dict | None:
    # ── 1. label_nutrition_mapping.json ──
    try:
        from detector.ml_food_predictor import get_nutrition_from_model_db
        result = get_nutrition_from_model_db(food_name)
        if result:
            return result
    except Exception as e:
        print(f"[NUTRITION] label_nutrition_mapping error: {e}")

    # ── 2. Spoonacular API ──
    if getattr(settings, "USE_API_NUTRITION", False):
        try:
            from services.api_nutrition import get_nutrition_from_api
            result = get_nutrition_from_api(food_name)
            if result:
                return result
        except Exception as e:
            print(f"[NUTRITION] API error: {e}")

    # ── 3. CSV fallback ──
    try:
        from services.csv_nutrition import get_nutrition_from_csv
        return get_nutrition_from_csv(food_name)
    except Exception as e:
        print(f"[NUTRITION] CSV error: {e}")

    return None
```

---

## Step 5 — Show Model Badge in `templates/result.html`

Find the confidence badge section and add the model source badge right after it:

```html
<!-- FIND THIS BLOCK: -->
{% if needs_confirmation %}
<div class="confidence-badge--low">⚠ Low Confidence ({{ confidence }}%)</div>
{% else %}
<div class="confidence-badge--high">✓ High Confidence ({{ confidence }}%)</div>
{% endif %}

<!-- REPLACE WITH: -->
{% if needs_confirmation %}
<div class="confidence-badge--low">⚠ Low Confidence ({{ confidence }}%)</div>
{% else %}
<div class="confidence-badge--high">✓ High Confidence ({{ confidence }}%)</div>
{% endif %}

<!-- Model source badge -->
{% if detection_method == "CUSTOM" %}
<span class="mm-pill mm-pill--good" style="margin-left:8px;font-size:11px;">
  🧠 Custom Model
</span>
{% elif detection_method == "CLIP" %}
<span class="mm-pill mm-pill--info" style="margin-left:8px;font-size:11px;">
  🔍 CLIP Fallback
</span>
{% elif detection_method %}
<span class="mm-pill" style="margin-left:8px;font-size:11px;">
  {{ detection_method }}
</span>
{% endif %}
```

---

## Step 6 — Show Model Badge in `templates/multi_result.html`

Inside the `{% for item in foods %}` loop, after the confidence badge:

```html
<!-- FIND: -->
{% if item.confidence %}
<span class="confidence-badge--{% if item.confidence >= 80 %}high{% else %}low{% endif %}">
  {% if item.confidence >= 80 %}✓{% else %}⚠{% endif %}
  {{ item.confidence }}% confidence
</span>
{% endif %}

<!-- ADD AFTER: -->
{% if item.detection_method == "CUSTOM" %}
<span class="mm-pill mm-pill--good" style="margin-top:6px;display:inline-flex;font-size:10px;">
  🧠 Custom
</span>
{% elif item.detection_method == "CLIP" %}
<span class="mm-pill mm-pill--info" style="margin-top:6px;display:inline-flex;font-size:10px;">
  🔍 CLIP
</span>
{% endif %}
```

---

## Step 7 — Update `detector/views.py` — pass detection_method to templates

In `upload_food` view, find where the result dict is passed to `result.html`
for **single image** (both high and low confidence paths) and add `detection_method`:

```python
# In the HIGH CONFIDENCE single image block, find:
"detection_method":   r.get("detection_method", "UNKNOWN"),

# Verify this line already exists. If not, add it to the context dict.
# It should already be there from the previous macromate-hybrid-model skill.
```

For the `_render_multi_result` helper, verify each result dict in the `results` list
contains `detection_method`. In `upload_food`, after `ml_result = predict_food(...)`:

```python
results.append({
    ...existing fields...,
    "detection_method": ml_result.get("method", "UNKNOWN"),
    "method_label":     ml_result.get("method_label", "AI"),
})
```

---

## Step 8 — Remove old model references

In `detector/ml_food_predictor.py` (already replaced in Step 3), there are no more
references to `final_model.h5`, `class_indices.json`, or `nutrition_lookup.json`.

You can safely delete the old files OR keep them (they won't be loaded):
```
models/final_model.h5       ← can delete (large file)
models/class_indices.json   ← can delete
models/nutrition_lookup.json ← can delete
```

---

## Expected Terminal Output After Integration

### Server startup (both loaded):
```
[MODEL LOADER] class_names.json loaded: 372 classes
[MODEL LOADER]   Sample classes: ['apple_pie', 'baby_back_ribs', 'baklava', ...]
[MODEL LOADER] label_nutrition_mapping.json loaded: 400 foods
[MODEL LOADER] Found best_model.pt: 214.0 MB
[MODEL LOADER] Loading PyTorch model on cpu...
[MODEL LOADER] Expected output classes: 372
[MODEL LOADER] Loaded as full saved model    ← or state_dict if needed
[MODEL LOADER] ✓ Custom PyTorch model ready
[MODEL LOADER] ✓ CLIP loaded on cpu
```

### When custom model wins:
```
────────────────────────────────────────────────────────────
[PREDICTOR] Image: samosa.jpg
[CUSTOM MODEL] ✓ Samosa (87.4%)
────────────────────────────────────────────────────────────
```

### When CLIP takes over:
```
────────────────────────────────────────────────────────────
[PREDICTOR] Image: blurry_food.jpg
[CUSTOM MODEL] Low confidence: Hamburger (38.1%) — trying CLIP
[CLIP FALLBACK] ✓ Hamburger (74.2%) — custom model low confidence
────────────────────────────────────────────────────────────
```

### When custom model not placed yet:
```
[MODEL LOADER] MISSING: D:\MacroMate\models\best_model.pt
[MODEL LOADER] Custom model unavailable — CLIP will handle all predictions
[MODEL LOADER] ✓ CLIP loaded on cpu
```

---

## Troubleshooting

### `[MODEL LOADER] All load strategies failed`

This means the `.pt` file architecture doesn't match EfficientNet-B0 or B3.
Find out what architecture was used for training by checking your training notebook.
Then update Strategy 2 in `_load_custom_model()` to use the correct architecture:

```python
# For ResNet50:
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# For EfficientNet-B4:
model = models.efficientnet_b4(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# For MobileNetV3:
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
```

### `class_names.json` format check

The loader handles both formats automatically:
```json
// Format 1 — list (preferred):
["apple_pie", "baby_back_ribs", "baklava", ...]

// Format 2 — dict with string keys:
{"0": "apple_pie", "1": "baby_back_ribs", ...}

// Format 3 — dict with int keys:
{0: "apple_pie", 1: "baby_back_ribs", ...}
```

### `label_nutrition_mapping.json` key format check

The lookup handles both underscore and space formats:
```json
{
  "apple_pie": {"calories": 237, "protein": 1.9, ...},
  "apple pie": {"calories": 237, ...}
}
```

### Confidence too low / always using CLIP

Lower the threshold in `ml_food_predictor.py`:
```python
CUSTOM_CONFIDENCE_THRESHOLD = 0.40   # was 0.50, try 0.40
```

### Wrong food predicted

Check top-5 in terminal and compare with `class_names.json`. If the model output
index doesn't match the class list order, the class_names.json may be in a different
order than training. Verify by checking your training script's class list.

---

## Decision Tree

```
Integrating new model?
└── Step 1: Place 3 files in models/ folder
└── Step 2: pip install torch torchvision (if not installed)
└── Step 3: Replace ml_food_predictor.py
└── Step 4: Update nutrition_provider.py
└── Step 5-6: Add model badges to result.html and multi_result.html
└── Step 7: Verify detection_method passed in views.py

Server starts, model loads, but predictions seem wrong?
└── Check CLASS_NAMES sample in terminal — does order match training?
└── Check: python manage.py shell
          from detector.ml_food_predictor import CLASS_NAMES
          print(CLASS_NAMES[:10])

label_nutrition_mapping.json — food not found?
└── Check actual key format in the JSON file
└── The partial match fallback will still find close matches
└── For complete misses, the CSV fallback activates automatically

best_model.pt loads but outputs nonsense / wrong food always?
└── Preprocessing mismatch — your model may use different normalization
└── Try changing Normalize values in _preprocess_pytorch():
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]  ← try this if ImageNet doesn't work
└── Or check your training script's data transforms
```

---

## Important Notes

1. **`best_model.pt` is a PyTorch file** — loaded with `torch.load()`, NOT `tf.keras.models.load_model()`. The old TF loading code is completely removed.

2. **3 load strategies are tried automatically** — full saved model → EfficientNet-B0 state_dict → EfficientNet-B3 state_dict. The first one that works wins.

3. **`label_nutrition_mapping.json` replaces both `nutrition_lookup.json` and is the new Priority 1** — it's larger (103 KB vs 30 KB) and has more complete data.

4. **`class_names.json` must be in the same order as training** — the index of the class in this list must match the output neuron index from training. If the model always predicts the wrong food, this is the likely cause.

5. **Both CLIP and the custom model use PyTorch** — they share the same `torch` install. No TensorFlow needed anymore.

6. **Start server with** (Windows PowerShell):
   ```
   python manage.py runserver
   ```
   No special env vars needed — PyTorch doesn't require `TF_USE_LEGACY_KERAS`.
