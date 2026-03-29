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
