#!/usr/bin/env python
"""
Script to add test meal data for demonstration
"""
import os
import sys
import django
from datetime import datetime, timedelta

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'food_calorie_project.settings')
django.setup()

from detector.models import MealLog

def add_test_meals():
    """Add some test meals for today"""
    
    # Clear existing meals for today (for clean demo)
    today_meals = MealLog.get_todays_meals()
    print(f"Clearing {today_meals.count()} existing meals for today...")
    today_meals.delete()
    
    # Sample meals with realistic times throughout the day
    test_meals = [
        {"food_name": "Pizza", "calories": 266, "hours_ago": 1},
        {"food_name": "Hamburger", "calories": 295, "hours_ago": 3},
        {"food_name": "Sushi", "calories": 130, "hours_ago": 5},
        {"food_name": "French Fries", "calories": 365, "hours_ago": 6},
        {"food_name": "Ice Cream", "calories": 207, "hours_ago": 8},
    ]
    
    created_meals = []
    for meal_data in test_meals:
        # Create meal with specific time
        meal_time = datetime.now() - timedelta(hours=meal_data["hours_ago"])
        
        meal = MealLog(
            food_name=meal_data["food_name"],
            calories=meal_data["calories"],
        )
        meal.save()
        
        # Manually update the created_at time (for demo purposes)
        meal.created_at = meal_time
        meal.save()
        
        created_meals.append(meal)
        print(f"✅ Added: {meal.food_name} - {meal.calories} kcal at {meal.get_time_display()}")
    
    # Calculate totals
    total_calories = sum(meal.calories for meal in created_meals)
    print(f"\n📊 Total: {len(created_meals)} meals, {total_calories} calories")
    print(f"🎯 Average: {total_calories // len(created_meals)} calories per meal")
    
    return created_meals

if __name__ == "__main__":
    print("🍽️ Adding test meal data...")
    meals = add_test_meals()
    print(f"\n✅ Successfully added {len(meals)} test meals!")
    print("🌐 Visit http://127.0.0.1:8000/history/ to see the meal history")