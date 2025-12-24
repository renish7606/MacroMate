import requests
from django.conf import settings

# Spoonacular endpoints
SEARCH_ENDPOINT = "https://api.spoonacular.com/food/ingredients/search"
INFO_ENDPOINT = "https://api.spoonacular.com/food/ingredients/{id}/information"

# Default portion sizes in grams
DEFAULT_PORTIONS = {
    "pizza": 300,        # 1 slice
    "burger": 250,       # 1 burger
    "rice": 200,         # 1 bowl
    "ice cream": 100,    # 1 scoop
    "bread": 30,         # 1 slice
    "apple": 180,        # 1 medium apple
    "banana": 120,
    "egg": 60,
}


def get_default_grams(food_name: str) -> int:
    """
    Return default grams for a food.
    Falls back to 100g if food not listed.
    """
    return DEFAULT_PORTIONS.get(food_name.lower(), 100)


def get_nutrition_from_api(food_name, grams=None):
    print("🔥 CALLING SPOONACULAR API 🔥")

    # ✅ ALWAYS define grams first
    if grams is None:
        grams = get_default_grams(food_name)

    try:
        # -------- Step 1: Search ingredient --------
        search_params = {
            "query": food_name,
            "number": 1,
            "apiKey": settings.SPOONACULAR_API_KEY,
        }

        search_resp = requests.get(
            SEARCH_ENDPOINT,
            params=search_params,
            timeout=5
        )

        if search_resp.status_code != 200:
            print("❌ Spoonacular search error:", search_resp.text)
            return None

        results = search_resp.json().get("results", [])
        if not results:
            print("⚠️ No ingredient found")
            return None

        ingredient_id = results[0]["id"]

        # -------- Step 2: Nutrition for EXACT grams --------
        info_params = {
            "amount": grams,   # ✅ ALWAYS defined
            "unit": "g",
            "apiKey": settings.SPOONACULAR_API_KEY,
        }

        info_resp = requests.get(
            INFO_ENDPOINT.format(id=ingredient_id),
            params=info_params,
            timeout=5
        )

        if info_resp.status_code != 200:
            print("❌ Spoonacular nutrition error:", info_resp.text)
            return None

        nutrients = info_resp.json().get("nutrition", {}).get("nutrients", [])

        def find(name):
            for n in nutrients:
                if n["name"].lower() == name.lower():
                    return round(n["amount"], 1)
            return 0.0

        return {
            "calories": find("Calories"),
            "protein": find("Protein"),
            "carbs": find("Carbohydrates"),
            "fat": find("Fat"),
        }

    except Exception as e:
        print("❌ Spoonacular exception:", e)
        return None
