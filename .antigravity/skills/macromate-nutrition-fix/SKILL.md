---
name: macromate-exact-fix
description: >
  MacroMate exact fix skill. Use this to fix two confirmed bugs visible from
  training screenshots: (1) nutrition shows 0 because label_nutrition_mapping.json
  stores data nested under "nutrition_data" key but code reads top-level fields,
  and (2) custom model (EfficientNetV2-M, 351 classes) loads with wrong architecture
  (B0/B3) causing garbage predictions which fall back to CLIP. This skill rewrites
  only the exact broken parts of ml_food_predictor.py — the model loader to use
  EfficientNetV2-M and the nutrition extractor to read from the nested nutrition_data dict.
---

# MacroMate Exact Fix Skill

## What the Screenshots Confirmed

### JSON Format (from image 6, lines 392–403)
```json
"almond jelly": {
    "nutrition_label": "almond oil",
    "nutrition_data": {
        "original_name": "almond oil",
        "calories": 120.0,
        "protein": 0.0,
        "carbohydrates": 0.0,
        "fat": 13.6,
        "fiber": 0.0,
        "sugar": 0.0
    },
    "aligned": true
}
```

**The nutrition is inside `nutrition_data` → but `_extract_nutrition()` reads
`entry["calories"]` which doesn't exist at the top level → always returns None → 0.**

### Model Architecture (from image 1)
- Architecture: **EfficientNetV2-M**
- Classes: **351**
- Parameters: **54,529,875**
- Pizza confidence in standalone script: **94.6%** (image 6)
- Hamburger confidence in standalone script: **82.2%** (images 7–10)

**But the Django loader tries EfficientNet-B0 then B3 — different architecture →
wrong weights loaded → garbage confidence → falls back to CLIP.**

---

## Fix 1 — Model Loader: Use EfficientNetV2-M

In `detector/ml_food_predictor.py`, replace the entire loading section inside
`_load_custom_model()` where it tries to load the PyTorch model:

```python
# ── 3. best_model.pt ──
if not os.path.exists(PYTORCH_MODEL_PATH):
    print(f"[MODEL LOADER] MISSING: {PYTORCH_MODEL_PATH}")
    print(f"[MODEL LOADER] CLIP will handle all predictions")
    return

size_mb = os.path.getsize(PYTORCH_MODEL_PATH) / (1024 * 1024)
print(f"[MODEL LOADER] Found best_model.pt: {size_mb:.1f} MB")

try:
    import torch
    import torchvision.models as models
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(CLASS_NAMES) if CLASS_NAMES else 351

    print(f"[MODEL LOADER] Loading EfficientNetV2-M on {device}...")
    print(f"[MODEL LOADER] Expected classes: {num_classes}")

    # ── STRATEGY 1: Full saved model (torch.save(model, path)) ──
    try:
        m = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=False)
        m.eval()
        custom_model = m
        print(f"[MODEL LOADER] ✓ Loaded as full saved model (Strategy 1)")
        print(f"[MODEL LOADER]   Type: {type(custom_model).__name__}")
        print(f"{'─'*50}")
        return
    except Exception as e1:
        print(f"[MODEL LOADER] Strategy 1 (full model) failed: {e1}")

    # ── STRATEGY 2: EfficientNetV2-M state_dict ──
    # This matches your training: EfficientNetV2-M, 54.5M params, 351 classes
    try:
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

        state = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True)

        # Handle checkpoint wrappers
        if isinstance(state, dict):
            for wrapper_key in ["model_state_dict", "state_dict", "model", "net"]:
                if wrapper_key in state:
                    print(f"[MODEL LOADER] Unwrapping checkpoint key: '{wrapper_key}'")
                    state = state[wrapper_key]
                    break

        model.load_state_dict(state, strict=True)
        model = model.to(device)
        model.eval()
        custom_model = model
        print(f"[MODEL LOADER] ✓ Loaded as EfficientNetV2-M state_dict (Strategy 2)")
        print(f"{'─'*50}")
        return
    except Exception as e2:
        print(f"[MODEL LOADER] Strategy 2 (EfficientNetV2-M strict) failed: {e2}")

    # ── STRATEGY 3: EfficientNetV2-M state_dict, strict=False ──
    # Handles minor key mismatches (e.g. BatchNorm running_mean buffer)
    try:
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

        state = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True)
        if isinstance(state, dict):
            for wrapper_key in ["model_state_dict", "state_dict", "model", "net"]:
                if wrapper_key in state:
                    state = state[wrapper_key]
                    break

        missing, unexpected = model.load_state_dict(state, strict=False)
        model = model.to(device)
        model.eval()
        custom_model = model
        print(f"[MODEL LOADER] ✓ Loaded as EfficientNetV2-M state_dict strict=False (Strategy 3)")
        if missing:
            print(f"[MODEL LOADER]   Missing keys: {len(missing)}")
        if unexpected:
            print(f"[MODEL LOADER]   Unexpected keys: {len(unexpected)}")
        print(f"{'─'*50}")
        return
    except Exception as e3:
        print(f"[MODEL LOADER] Strategy 3 (EfficientNetV2-M loose) failed: {e3}")
        print(f"[MODEL LOADER] All strategies failed. CLIP will handle predictions.")
        print(f"[MODEL LOADER]   Error details:")
        print(f"[MODEL LOADER]   S1: {e1}")
        print(f"[MODEL LOADER]   S2: {e2}")
        print(f"[MODEL LOADER]   S3: {e3}")
        custom_model = None

except ImportError as e:
    print(f"[MODEL LOADER] Import error: {e}")
    print(f"[MODEL LOADER] Run: pip install torch torchvision")
    custom_model = None
except Exception as e:
    print(f"[MODEL LOADER] Unexpected error: {e}")
    custom_model = None
```

---

## Fix 2 — Preprocessing: EfficientNetV2-M Uses Different Input Size

EfficientNetV2-M uses **480×480** input (not 224×224 like B0/B3).
Replace `_preprocess_pytorch()`:

```python
def _preprocess_pytorch(image_path: str):
    """
    Preprocess for EfficientNetV2-M.
    Input size: 480x480 (NOT 224 — that's B0/B3)
    Normalization: ImageNet mean/std
    """
    import torch
    from torchvision import transforms

    # EfficientNetV2-M recommended input size is 480
    preprocess = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(480),
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
```

---

## Fix 3 — Nutrition Extractor: Read from `nutrition_data` Nested Key

This is the **root cause of all 0 values**. Replace `_extract_nutrition()`:

```python
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

    The nutrition values are INSIDE "nutrition_data" — not at the top level.
    """
    if entry is None or not isinstance(entry, dict):
        return None

    # ── Drill into nutrition_data if present (confirmed JSON structure) ──
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
```

---

## Fix 4 — Confidence Thresholds

Your model gets 94.6% for pizza and 82.2% for hamburger in the standalone script.
In Django with torchvision transforms, it will be slightly lower but still high.
Update the constants at the top of `ml_food_predictor.py`:

```python
# ─────────────────────────────────────────
# THRESHOLDS
# EfficientNetV2-M, 351 classes
# Standalone script shows 82-94% for common foods → set win threshold at 40%
# ─────────────────────────────────────────
CUSTOM_WIN_THRESHOLD       = 40.0   # custom model wins outright at 40%+
CUSTOM_CONSIDER_THRESHOLD  = 20.0   # wins at 20%+ if food is in nutrition DB
CLIP_CONFIDENCE_THRESHOLD  = 25.0
IMG_SIZE                   = 480    # EfficientNetV2-M uses 480, not 224
```

---

## Fix 5 — Nutrition Loader: Handle Nested JSON at Load Time

Replace the nutrition loading block in `_load_custom_model()`:

```python
# ── 2. label_nutrition_mapping.json ──
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

        # Verify known foods — these MUST be in the DB
        for test_food in ["pizza", "hamburger", "ice cream", "samosa", "sushi"]:
            if test_food in NUTRITION_DB:
                # Also verify nutrition_data is accessible
                entry = NUTRITION_DB[test_food]
                nd = entry.get("nutrition_data", {})
                cal = nd.get("calories", 0)
                print(f"[MODEL LOADER]   ✓ '{test_food}': {cal} kcal")
            else:
                print(f"[MODEL LOADER]   ✗ '{test_food}': NOT IN DB")

    except Exception as e:
        print(f"[MODEL LOADER] label_nutrition_mapping load error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[MODEL LOADER] MISSING: {NUTRITION_MAP_PATH}")
```

---

## Fix 6 — Quick Verification After Restart

After applying all fixes and restarting, you should see in terminal:

```
[MODEL LOADER] label_nutrition_mapping loaded: 351 entries
[MODEL LOADER]   ✓ 'pizza': 308 kcal
[MODEL LOADER]   ✓ 'hamburger': 255 kcal
[MODEL LOADER]   ✓ 'ice cream': ... kcal
[MODEL LOADER] Found best_model.pt: 214.0 MB
[MODEL LOADER] Loading EfficientNetV2-M on cpu...
[MODEL LOADER] ✓ Loaded as EfficientNetV2-M state_dict (Strategy 2)
[MODEL LOADER] ✓ CLIP loaded on cpu
```

When uploading pizza:
```
────────────────────────────────────────────────────────────
[PREDICTOR] Image: pizza.jpg
[CUSTOM MODEL] Pizza | conf=92.4% | in_nutrition_db=True
[CUSTOM MODEL] ✓ Won (high confidence 92.4% >= 40.0%)
[NUTRITION DB] ✓ Exact match: 'pizza' → 308 kcal
────────────────────────────────────────────────────────────
```

---

## Summary of All Changes

| File | What changes |
|---|---|
| `ml_food_predictor.py` | Model loader uses EfficientNetV2-M (not B0/B3) |
| `ml_food_predictor.py` | `_preprocess_pytorch()` uses 480px input (not 224) |
| `ml_food_predictor.py` | `_extract_nutrition()` reads `entry["nutrition_data"]` first |
| `ml_food_predictor.py` | Nutrition loader verifies known foods at startup |
| `ml_food_predictor.py` | `CUSTOM_WIN_THRESHOLD` = 40% (was 50%) |
| `ml_food_predictor.py` | `IMG_SIZE` = 480 (was 224) |

**Only one file changes: `detector/ml_food_predictor.py`**

---

## Decision Tree

```
Still shows CLIP after fix?
├── Check terminal for "[MODEL LOADER] ✓ Loaded as EfficientNetV2-M"
│   └── If not shown → Strategy 1 (full saved model) may be the right one
│       Check: does your training script use torch.save(model, path)?
│       If yes → Strategy 1 should work, check what error Strategy 2 shows
│
└── Check: "[CUSTOM MODEL] Pizza | conf=XX%"
    └── If conf < 40% → the model loaded wrong architecture
        → Verify best_model.pt is the new model (214 MB), not old one

Still 0 nutrition after fix?
├── Check terminal for "[NUTRITION DB] ✓ 'pizza': 308 kcal" at startup
│   └── If shows 0 → JSON key for pizza is not "pizza" — check actual key:
│       python manage.py shell
│       from detector.ml_food_predictor import NUTRITION_DB
│       [k for k in NUTRITION_DB if "pizza" in k]
│
└── Check: "[NUTRITION DB] ✓ Exact match: 'pizza' → 308 kcal" at upload time
    └── If not shown → food name from model doesn't match JSON key
        The class_names.json has "pizza" but JSON key might be "Pizza" or "pizza_plain"
```
