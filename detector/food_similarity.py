import csv
import os
from difflib import get_close_matches
from django.conf import settings

# --------------------------------------------------
# PATH
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALORIE_PATH = os.path.join(BASE_DIR, "data", "calories.csv")

CSV_FOODS = []
FOOD_CALORIES = {}
FOOD_NUTRITION = {}

# --------------------------------------------------
# LOAD CSV FOODS (SINGLE SOURCE OF TRUTH)
# --------------------------------------------------
def load_csv_foods():
    csv_path = os.path.join(settings.BASE_DIR, "data", "calories.csv")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            food = row["food"].strip().lower().replace(" ", "_")

            FOOD_NUTRITION[food] = {
                "calories": int(row["calorie"]),
                "protein": float(row["protein"]),
                "carbs": float(row["carbs"]),
                "fat": float(row["fat"]),
            }

            CSV_FOODS.append(food)

# load once
load_csv_foods()


def get_nutrition_for_food(food):
    food = food.lower().replace(" ", "_")
    return FOOD_NUTRITION.get(food)


def get_related_foods(food):
    food = food.lower().replace(" ", "_")
    related = []

    for f in CSV_FOODS:
        if f != food and (food in f or f in food):
            related.append(f.replace("_", " ").title())

    return related[:5]