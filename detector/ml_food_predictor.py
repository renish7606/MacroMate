"""
MacroMate Hybrid Food Detector v2
------------------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------------------------------------------------------
BASE_DIR              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH    = os.path.join(BASE_DIR, "models", "best_model.pt")
PYTORCH_MODEL_PATH    = os.getenv("MM_MODEL_PATH", DEFAULT_MODEL_PATH)
CLASS_NAMES_PATH      = os.path.join(BASE_DIR, "models", "class_names.json")
NUTRITION_MAP_PATH    = os.path.join(BASE_DIR, "models", "label_nutrition_mapping.json")

# ---------------------------------------------------------------------------------------------------------------------------
# THRESHOLDS
# EfficientNetV2-M, 351 classes
# Standalone script shows 82-94% for common foods --- set win threshold at 40%
# ---------------------------------------------------------------------------------------------------------------------------
CUSTOM_WIN_THRESHOLD       = 40.0   # custom model wins outright at 40%+
CUSTOM_CONSIDER_THRESHOLD  = 20.0   # wins at 20%+ if food is in nutrition DB
CLIP_CONFIDENCE_THRESHOLD  = 25.0
CUSTOM_OOD_FALLBACK_THRESHOLD = 5.0  # Only below this we consider "outside custom classes"
IMG_SIZE                   = 480    # EfficientNetV2-M uses 480, not 224

# ---------------------------------------------------------------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------------------------------------------------------------
custom_model   = None      # PyTorch model
CLASS_NAMES    = []        # ["apple_pie", "baby_back_ribs", ...]
NUTRITION_DB   = {}        # {"apple_pie": {calories, protein, ...}}
ACTIVE_MODEL_PATH = None

# Canonical label mapping to reduce near-duplicate class name variance.
CLASS_ALIASES = {
    # Pizza family
    "margherita_pizza": "pizza",
    "pepperoni_pizza": "pizza",
    "cheese_pizza": "pizza",
    "veggie_pizza": "pizza",
    "pizza_slice": "pizza",
    # Sushi family
    "spam_musubi": "sushi",
    "maki": "sushi",
    "sushi_roll": "sushi",
    "california_roll": "sushi",
    "nigiri": "sushi",
    "sashimi": "sushi",
    # Ice cream family
    "ice_cream_cone": "ice_cream",
    "gelato": "ice_cream",
    "frozen_yogurt": "ice_cream",
}


def _canonical_food_key(food_name: str) -> str:
    """
    Normalize predicted label into canonical key used by app/nutrition.
    """
    key = (food_name or "").strip().lower().replace(" ", "_")
    # Reject obviously bad class labels from noisy class_names entries.
    if key.isdigit():
        return "unknown"
    return CLASS_ALIASES.get(key, key)


def _is_suspicious_label(food_key: str) -> bool:
    key = (food_key or "").strip().lower()
    if not key:
        return True
    if key == "unknown":
        return True
    if key.isdigit():
        return True
    return False


def _resolve_model_path():
    """
    Pick the first existing model path from:
    1) MM_MODEL_PATH env var
    2) project default path
    3) common local training output paths
    """
    env_path = os.getenv("MM_MODEL_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(env_path)

    candidates.extend([
        DEFAULT_MODEL_PATH,
        os.path.join(os.path.dirname(BASE_DIR), "macromate-ml-training", "best_model.pt"),
        os.path.join(os.path.dirname(BASE_DIR), "macromate-ml-training", "models", "best_model.pt"),
        os.path.join("C:\\", "macromate-ml-training", "best_model.pt"),
        os.path.join("C:\\", "macromate-ml-training", "models", "best_model.pt"),
    ])

    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.exists(norm):
            return norm
    return None


# ---------------------------------------------------------------------------------------------------------------------------
# LOADER --- called ONCE at Django startup
# ---------------------------------------------------------------------------------------------------------------------------

def _unwrap_checkpoint_state(state):
    """
    Extract state_dict from common checkpoint wrapper keys.
    """
    if not isinstance(state, dict):
        return state

    for wrapper_key in ["model_state_dict", "state_dict", "model", "net", "ema_state_dict", "model_state"]:
        value = state.get(wrapper_key)
        if isinstance(value, dict):
            print(f"[MODEL LOADER] Unwrapping checkpoint key: '{wrapper_key}'")
            return value
    return state


def _strip_common_prefixes(state_dict: dict):
    """
    Strip common prefix wrappers such as DataParallel 'module.'.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    for prefix in ["module.", "model.", "net.", "_orig_mod."]:
        if all(k.startswith(prefix) for k in keys):
            print(f"[MODEL LOADER] Stripping key prefix: '{prefix}'")
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _configure_classifier_from_state(model, state, nn, fallback_num_classes: int):
    """
    Build classifier head to match checkpoint layout.
    Supports:
    - default torchvision head: classifier.1 (single linear)
    - custom head: classifier.1 + classifier.4 (1280 -> hidden -> num_classes)
    """
    if not isinstance(state, dict):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, fallback_num_classes)
        return model

    if "classifier.4.weight" in state and "classifier.1.weight" in state:
        hidden_dim = int(state["classifier.1.weight"].shape[0])
        in_dim = int(state["classifier.1.weight"].shape[1])
        out_dim = int(state["classifier.4.weight"].shape[0])
        print(f"[MODEL LOADER] Detected custom classifier head: {in_dim} -> {hidden_dim} -> {out_dim}")
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        return model

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, fallback_num_classes)
    return model


def _model_weights_look_healthy(model) -> tuple[bool, str]:
    """
    Basic checkpoint sanity check:
    - no NaN / Inf
    - no extreme magnitudes that cause overflow at inference
    """
    import torch

    max_abs = 0.0
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not torch.is_floating_point(tensor):
            continue
        if torch.isnan(tensor).any():
            return False, f"NaN values in tensor '{name}'"
        if torch.isinf(tensor).any():
            return False, f"Inf values in tensor '{name}'"

        local_max = float(torch.max(torch.abs(tensor)).item())
        if local_max > max_abs:
            max_abs = local_max
        if local_max > 1e6:
            return False, f"Extreme magnitude in tensor '{name}' (max_abs={local_max:.3e})"

    return True, f"max_abs={max_abs:.3e}"
def _load_custom_model():
    global custom_model, CLASS_NAMES, NUTRITION_DB, ACTIVE_MODEL_PATH

    # ------ 1. class_names.json ------
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

    # ------ 2. label_nutrition_mapping.json ------
    if os.path.exists(NUTRITION_MAP_PATH):
        try:
            with open(NUTRITION_MAP_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)

            NUTRITION_DB.clear()

            if isinstance(raw, dict):
                # Format: {"pizza": {"nutrition_data": {...}, "aligned": true}, ...}
                for k, v in raw.items():
                    norm_key = str(k).strip().lower().replace("_", " ")
                    NUTRITION_DB[norm_key] = v

            elif isinstance(raw, list):
                # Format: [{"label": "pizza", "nutrition_data": {...}}, ...]
                for item in raw:
                    if isinstance(item, dict):
                        label = (
                            item.get("label") or
                            item.get("name") or
                            item.get("food") or
                            item.get("class_name") or ""
                        )
                        norm_key = str(label).strip().lower().replace("_", " ")
                        if norm_key:
                            NUTRITION_DB[norm_key] = item

            print(f"[MODEL LOADER] label_nutrition_mapping loaded: {len(NUTRITION_DB)} entries")

            # Verify known foods --- these MUST be in the DB
            for test_food in ["pizza", "hamburger", "ice cream", "samosa", "sushi"]:
                if test_food in NUTRITION_DB:
                    # Also verify nutrition_data is accessible
                    entry = NUTRITION_DB[test_food]
                    nd = entry.get("nutrition_data", {})
                    cal = nd.get("calories", 0)
                    print(f"[MODEL LOADER]   --- '{test_food}': {cal} kcal")
                else:
                    print(f"[MODEL LOADER]   --- '{test_food}': NOT IN DB")

        except Exception as e:
            print(f"[MODEL LOADER] label_nutrition_mapping load error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[MODEL LOADER] MISSING: {NUTRITION_MAP_PATH}")

    # ------ 3. best_model.pt ------
    ACTIVE_MODEL_PATH = _resolve_model_path()
    if not ACTIVE_MODEL_PATH:
        print(f"[MODEL LOADER] MISSING: {PYTORCH_MODEL_PATH}")
        print("[MODEL LOADER] Tip: set MM_MODEL_PATH to your working best_model.pt")
        print(f"[MODEL LOADER] CLIP will handle all predictions")
        return

    size_mb = os.path.getsize(ACTIVE_MODEL_PATH) / (1024 * 1024)
    print(f"[MODEL LOADER] Found best_model.pt: {size_mb:.1f} MB")
    print(f"[MODEL LOADER] Path: {ACTIVE_MODEL_PATH}")

    try:
        import torch
        import torchvision.models as models
        import torch.nn as nn

        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = len(CLASS_NAMES) if CLASS_NAMES else 351

        print(f"[MODEL LOADER] Loading EfficientNetV2-M on {device}...")
        print(f"[MODEL LOADER] Expected classes: {num_classes}")
        s1_error = None
        s2_error = None

        # ------ STRATEGY 1: Full saved model (torch.save(model, path)) ------
        try:
            m = torch.load(ACTIVE_MODEL_PATH, map_location=device, weights_only=False)
            m.eval()
            ok, reason = _model_weights_look_healthy(m)
            if not ok:
                raise RuntimeError(f"Loaded full model is unhealthy: {reason}")
            custom_model = m
            print(f"[MODEL LOADER] --- Loaded as full saved model (Strategy 1)")
            print(f"[MODEL LOADER]   Type: {type(custom_model).__name__}")
            print(f"[MODEL LOADER]   Health: {reason}")
            print(f"{'---'*50}")
            return
        except Exception as e1:
            s1_error = e1
            print(f"[MODEL LOADER] Strategy 1 (full model) failed: {e1}")

        # ------ STRATEGY 2: EfficientNetV2-M state_dict ------
        # This matches your training: EfficientNetV2-M, 54.5M params, 351 classes
        try:
            model = models.efficientnet_v2_m(weights=None)

            state = torch.load(ACTIVE_MODEL_PATH, map_location=device, weights_only=True)
            state = _unwrap_checkpoint_state(state)
            state = _strip_common_prefixes(state)
            model = _configure_classifier_from_state(model, state, nn, num_classes)

            model.load_state_dict(state, strict=True)
            model = model.to(device)
            model.eval()
            ok, reason = _model_weights_look_healthy(model)
            if not ok:
                raise RuntimeError(f"Loaded state_dict is unhealthy: {reason}")
            custom_model = model
            print(f"[MODEL LOADER] --- Loaded as EfficientNetV2-M state_dict (Strategy 2)")
            print(f"[MODEL LOADER]   Health: {reason}")
            print(f"{'---'*50}")
            return
        except Exception as e2:
            s2_error = e2
            print(f"[MODEL LOADER] Strategy 2 (EfficientNetV2-M strict) failed: {e2}")

        # ------ STRATEGY 3: EfficientNetV2-M state_dict, strict=False ------
        # Handles minor key mismatches (e.g. BatchNorm running_mean buffer)
        try:
            model = models.efficientnet_v2_m(weights=None)

            state = torch.load(ACTIVE_MODEL_PATH, map_location=device, weights_only=True)
            state = _unwrap_checkpoint_state(state)
            state = _strip_common_prefixes(state)
            model = _configure_classifier_from_state(model, state, nn, num_classes)

            missing, unexpected = model.load_state_dict(state, strict=False)
            total_keys = len(model.state_dict())
            loaded_keys = total_keys - len(missing)
            coverage = (loaded_keys / total_keys * 100.0) if total_keys else 0.0

            if coverage < 85.0:
                raise RuntimeError(
                    f"Unsafe partial load: coverage={coverage:.1f}% "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )

            model = model.to(device)
            model.eval()
            ok, reason = _model_weights_look_healthy(model)
            if not ok:
                raise RuntimeError(f"Loaded loose state_dict is unhealthy: {reason}")
            custom_model = model
            print(f"[MODEL LOADER] --- Loaded as EfficientNetV2-M state_dict strict=False (Strategy 3)")
            print(f"[MODEL LOADER]   Coverage: {coverage:.1f}%")
            print(f"[MODEL LOADER]   Health: {reason}")
            if missing:
                print(f"[MODEL LOADER]   Missing keys: {len(missing)}")
            if unexpected:
                print(f"[MODEL LOADER]   Unexpected keys: {len(unexpected)}")
            print(f"{'---'*50}")
            return
        except Exception as e3:
            print(f"[MODEL LOADER] Strategy 3 (EfficientNetV2-M loose) failed: {e3}")
            print(f"[MODEL LOADER] All strategies failed. CLIP will handle predictions.")
            print(f"[MODEL LOADER]   Error details:")
            print(f"[MODEL LOADER]   S1: {s1_error}")
            print(f"[MODEL LOADER]   S2: {s2_error}")
            print(f"[MODEL LOADER]   S3: {e3}")
            custom_model = None

    except ImportError as e:
        print(f"[MODEL LOADER] Import error: {e}")
        print(f"[MODEL LOADER] Run: pip install torch torchvision")
        custom_model = None
    except Exception as e:
        print(f"[MODEL LOADER] Unexpected error: {e}")
        custom_model = None


# ---------------------------------------------------------------------------------------------------------------------------
# LOAD CLIP
# ---------------------------------------------------------------------------------------------------------------------------
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
    print(f"[MODEL LOADER] --- CLIP loaded on {clip_device}")
except ImportError:
    print("[MODEL LOADER] CLIP not installed --- pip install git+https://github.com/openai/CLIP.git")
except Exception as e:
    print(f"[MODEL LOADER] CLIP failed: {e}")

# ---------------------------------------------------------------------------------------------------------------------------
# CLIP FOOD LIST (comprehensive)
# ---------------------------------------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------------------------
# Run loaders at import time (once at startup)
# ---------------------------------------------------------------------------------------------------------------------------
_load_custom_model()


# ---------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING FOR PYTORCH MODEL
# ---------------------------------------------------------------------------------------------------------------------------
def _preprocess_pytorch(image_path: str):
    """
    Preprocess for EfficientNetV2-M.
    Input size: 480x480 (NOT 224 --- that's B0/B3)
    Normalization: ImageNet mean/std
    """
    import torch
    from torchvision import transforms

    # Force deterministic square resize to preserve full dish context.
    # CenterCrop can cut off pizza/sushi plate edges and hurt accuracy.
    preprocess = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    device = next(custom_model.parameters()).device
    return tensor.to(device)


# ---------------------------------------------------------------------------------------------------------------------------
# CUSTOM MODEL PREDICTION
# ---------------------------------------------------------------------------------------------------------------------------

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

        raw_food_name = CLASS_NAMES[best_idx]
        food_name = _canonical_food_key(raw_food_name)
        confidence = float(probs_np[best_idx])

        # Top-5 predictions
        top5_idx = np.argsort(probs_np)[-5:][::-1]
        top5 = []
        for i in top5_idx:
            if i >= len(CLASS_NAMES):
                continue
            raw_top = CLASS_NAMES[i]
            canonical_top = _canonical_food_key(raw_top)
            top5.append({
                "food": canonical_top.replace("_", " ").title(),
                "confidence": round(float(probs_np[i]) * 100, 2),
            })

        return {
            "food_name": food_name,
            "display_name": food_name.replace("_", " ").title(),
            "confidence": round(confidence * 100, 2),
            "is_confident": round(confidence * 100, 2) >= CUSTOM_WIN_THRESHOLD,
            "top_predictions": top5,
            "method": "CUSTOM",
            "method_label": "Custom Model",
        }

    except Exception as e:
        print(f"[CUSTOM MODEL] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _should_use_custom_result(result: dict | None) -> bool:
    """
    Decide whether custom model output should be final.
    Requirement: for foods in trained classes, prefer custom model.
    So CLIP is used only when custom confidence is extremely low.
    """
    if not result:
        return False

    if _is_suspicious_label(result.get("food_name", "")):
        return False

    confidence = float(result.get("confidence", 0.0))
    if confidence < CUSTOM_OOD_FALLBACK_THRESHOLD:
        return False

    # Strong custom confidence -> custom wins.
    if confidence >= CUSTOM_WIN_THRESHOLD:
        return True

    # Medium confidence accepted only if nutrition mapping exists.
    if confidence >= CUSTOM_CONSIDER_THRESHOLD:
        return get_nutrition_from_model_db(result.get("food_name", "")) is not None

    return False


# ---------------------------------------------------------------------------------------------------------------------------
# CLIP PREDICTION
# ---------------------------------------------------------------------------------------------------------------------------
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
            "is_confident":    best["confidence"] >= CLIP_CONFIDENCE_THRESHOLD,
            "top_predictions": results[:5],
            "method":          "CLIP",
            "method_label":    "CLIP Fallback",
        }

    except Exception as e:
        print(f"[CLIP] Prediction error: {e}")
        return None


# ---------------------------------------------------------------------------------------------------------------------------
# PUBLIC API --- called by views.py
# ---------------------------------------------------------------------------------------------------------------------------
def predict_food(image_path: str) -> dict:
    """
    Hybrid predictor.
    1. Try custom PyTorch model first.
    2. If custom is weak/unreliable, fall back to CLIP.
    """
    print(f"\n{'-'*60}")
    print(f"[PREDICTOR] Image: {os.path.basename(image_path)}")

    custom_result = None

    if custom_model is not None:
        custom_result = _predict_custom(image_path)

        if _should_use_custom_result(custom_result):
            print(f"[CUSTOM MODEL] OK {custom_result['display_name']} ({custom_result['confidence']}%)")
            print(f"{'-'*60}\n")
            return custom_result

        if custom_result:
            print(
                f"[CUSTOM MODEL] Low confidence: {custom_result['display_name']} "
                f"({custom_result['confidence']}%) -> trying CLIP"
            )
        else:
            print("[CUSTOM MODEL] Prediction failed -> trying CLIP")
    else:
        print("[CUSTOM MODEL] Not loaded -> using CLIP directly")

    if CLIP_AVAILABLE:
        clip_result = _predict_clip(image_path)
        if clip_result:
            reason = "custom model unavailable" if custom_model is None else "custom model low confidence"
            print(f"[CLIP FALLBACK] OK {clip_result['display_name']} ({clip_result['confidence']}%) - {reason}")
            print(f"{'-'*60}\n")
            return clip_result

    if custom_result:
        print(
            f"[PREDICTOR] CLIP unavailable/failed -> returning custom low-confidence result: "
            f"{custom_result['display_name']} ({custom_result['confidence']}%)"
        )
        print(f"{'-'*60}\n")
        return custom_result

    print("[PREDICTOR] X All models failed")
    print(f"{'-'*60}\n")
    return {
        "food_name": "unknown",
        "display_name": "Unknown",
        "confidence": 0.0,
        "is_confident": False,
        "top_predictions": [],
        "method": "NONE",
        "method_label": "No Model",
        "error": "All models failed",
    }


def _extract_nutrition(entry) -> dict | None:
    """
    Extract nutrition from label_nutrition_mapping.json entry.

    Actual JSON structure (confirmed from screenshots):
    {
        "nutrition_label": "pizza",
        "nutrition_data": {
            "original_name": "pizza",
            "calories": 308.0,
            "protein": 12.7,
            "carbohydrates": 36.0,
            "fat": 12.6,
            "fiber": 0.0,
            "sugar": 0.0
        },
        "aligned": true
    }

    The nutrition values are INSIDE "nutrition_data" --- not at the top level.
    """
    if entry is None or not isinstance(entry, dict):
        return None

    # ------ Drill into nutrition_data if present (confirmed JSON structure) ------
    data = entry.get("nutrition_data", entry)

    # If nutrition_data is also nested (e.g. double-wrapped), unwrap again
    if isinstance(data, dict) and "nutrition_data" in data:
        data = data["nutrition_data"]

    if not isinstance(data, dict):
        return None

    def g(names, default=0.0) -> float:
        """Try multiple field name variants."""
        for name in names:
            if name in data:
                try:
                    val = float(data[name])
                    return round(val, 1)
                except (TypeError, ValueError):
                    pass
        return default

    calories = g(["calories", "Calories", "kcal", "energy", "cal"])
    protein  = g(["protein", "Protein"])
    carbs    = g(["carbohydrates", "carbs", "Carbohydrates", "Carbs", "carbohydrate"])
    fat      = g(["fat", "Fat", "total_fat", "fats"])
    fiber    = g(["fiber", "Fiber", "dietary_fiber", "fibre"])
    sugar    = g(["sugar", "Sugar", "sugars"])
    vit_a    = g(["vitamin_a", "Vitamin_A", "vitaminA"])
    vit_c    = g(["vitamin_c", "Vitamin_C", "vitaminC"])
    calcium  = g(["calcium", "Calcium"])
    iron     = g(["iron", "Iron"])

    # If calories is still 0, the entry is not useful
    if calories == 0.0:
        return None

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


def get_nutrition_from_model_db(food_name: str) -> dict | None:
    """
    Look up nutrition from label_nutrition_mapping.json.
    Returns dict with: calories, protein, carbs, fat, fiber, sugar,
    vitamin_a, vitamin_c, calcium, iron --- or None if not found.
    Tries exact match, then underscore/space variants, then partial match.
    """
    if not NUTRITION_DB:
        return None

    # Normalize key
    key = food_name.lower().strip()
    key_underscored = key.replace(" ", "_")
    key_spaced      = key.replace("_", " ")

    # 1. Exact match
    for k in [key, key_underscored, key_spaced]:
        if k in NUTRITION_DB:
            result = _extract_nutrition(NUTRITION_DB[k])
            if result:
                print(f"[NUTRITION DB] --- Exact match: '{k}' --- {result['calories']} kcal")
                return result

    # 2. Partial match (substring)
    for db_key in NUTRITION_DB:
        db_clean = db_key.lower().replace("_", " ")
        food_clean = key_spaced
        if food_clean in db_clean or db_clean in food_clean:
            result = _extract_nutrition(NUTRITION_DB[db_key])
            if result:
                print(f"[NUTRITION DB] --- Partial match: '{db_key}' --- {result['calories']} kcal")
                return result

    print(f"[NUTRITION DB] --- No match for: '{food_name}'")
    return None


# Legacy compatibility
def softmax_with_temperature(logits, temperature=1.0):
    exp = np.exp(logits / temperature)
    return exp / np.sum(exp)




