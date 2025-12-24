import csv
import os
from django.conf import settings

CSV_PATH = os.path.join(settings.BASE_DIR, "data", "calories.csv")

def get_nutrition_from_csv(food_name):
    food_name = food_name.lower().replace(" ", "_")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["food"].lower().replace(" ", "_") == food_name:
                return {
                    "calories": int(row.get("calorie", 0)),
                    "protein": float(row.get("protein", 0)),
                    "carbs": float(row.get("carbs", 0)),
                    "fat": float(row.get("fat", 0)),
                }

    return None
