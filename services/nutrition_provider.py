from django.conf import settings
from services.api_nutrition import get_nutrition_from_api
from services.csv_nutrition import get_nutrition_from_csv

def get_nutrition(food_name):
    # Try API first if enabled
    if settings.USE_API_NUTRITION:
        nutrition = get_nutrition_from_api(food_name)

        if nutrition:
            return nutrition

        print("⚠️ API failed, falling back to CSV")

    # Fallback to CSV
    return get_nutrition_from_csv(food_name)
