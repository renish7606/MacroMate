import csv
import os
from difflib import get_close_matches

# --------------------------------------------------
# PATH
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALORIE_PATH = os.path.join(BASE_DIR, "data", "calories.csv")

# --------------------------------------------------
# LOAD CSV FOODS (SINGLE SOURCE OF TRUTH)
# --------------------------------------------------
def load_csv_foods():
    foods = {}
    with open(CALORIE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["food"].strip().lower()
            calories = float(row["calories_per_100g"])
            foods[name] = calories
    return foods

# Load once
FOOD_DB = load_csv_foods()
CSV_FOODS = set(FOOD_DB.keys())

# --------------------------------------------------
# CALORIE LOOKUP
# --------------------------------------------------
def get_calories_for_food(food_name, serving_factor=1.0):
    if not food_name:
        return None

    food = food_name.strip().lower()
    if food in FOOD_DB:
        return int(FOOD_DB[food] * serving_factor)

    return None

# --------------------------------------------------
# RELATED FOODS
# --------------------------------------------------
def get_related_foods(food_name, limit=5):
    if not food_name:
        return []

    food = food_name.strip().lower()
    matches = get_close_matches(food, CSV_FOODS, n=limit + 1, cutoff=0.6)
    return [m.title() for m in matches if m != food][:limit]
