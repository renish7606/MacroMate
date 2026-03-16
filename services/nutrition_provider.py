"""
Nutrition Provider
Priority 1: nutrition_lookup.json (richer — fiber, sugar, vitamins)
Priority 2: Spoonacular API        (if USE_API_NUTRITION is True)
Priority 3: calories.csv           (legacy fallback)
"""
from django.conf import settings


def get_nutrition(food_name: str) -> dict | None:
    """
    Returns nutrition dict with: calories, protein, carbs, fat.
    Optionally also: fiber, sugar, vitamin_a, vitamin_c, calcium, iron
    (available when source is nutrition_lookup.json).
    """
    # ── 1. nutrition_lookup.json (new, richer) ──
    try:
        from detector.ml_food_predictor import get_nutrition_from_model_db
        result = get_nutrition_from_model_db(food_name)
        if result:
            return result
    except Exception as e:
        print(f"[NUTRITION] nutrition_lookup.json error: {e}")

    # ── 2. Spoonacular API (optional) ──
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
