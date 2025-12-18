import requests
import os

# Use environment variable name, not the actual key
API_KEY = "f556ee703ec844b49421511b4cb1100b"  # Direct key for now, move to env later

def fetch_food_info(query):
    """
    Fetch food information from Spoonacular API
    Try multiple endpoints for better results
    """
    try:
        # First try food search endpoint (more comprehensive)
        url = "https://api.spoonacular.com/food/ingredients/search"
        params = {
            "query": query,
            "number": 1,
            "apiKey": API_KEY
        }
        
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("results"):
                ingredient = data["results"][0]
                
                # Get nutrition info for this ingredient
                nutrition_url = f"https://api.spoonacular.com/food/ingredients/{ingredient['id']}/information"
                nutrition_params = {
                    "amount": 100,
                    "unit": "grams",
                    "apiKey": API_KEY
                }
                
                nutrition_res = requests.get(nutrition_url, params=nutrition_params, timeout=10)
                if nutrition_res.status_code == 200:
                    nutrition_data = nutrition_res.json()
                    
                    # Extract calories from nutrition data
                    calories = None
                    if "nutrition" in nutrition_data and "nutrients" in nutrition_data["nutrition"]:
                        for nutrient in nutrition_data["nutrition"]["nutrients"]:
                            if nutrient["name"].lower() == "calories":
                                calories = nutrient["amount"]
                                break
                    
                    return {
                        "food_name": ingredient["name"],
                        "calories": calories,
                        "related": [ingredient["name"]]  # Simple fallback
                    }
        
        # Fallback to menu items search
        menu_url = "https://api.spoonacular.com/food/menuItems/search"
        menu_params = {
            "query": query,
            "number": 5,
            "apiKey": API_KEY
        }

        menu_res = requests.get(menu_url, params=menu_params, timeout=10)
        if menu_res.status_code == 200:
            menu_data = menu_res.json()
            
            if menu_data.get("menuItems"):
                item = menu_data["menuItems"][0]
                
                calories = None
                if "nutrition" in item and "calories" in item["nutrition"]:
                    calories = item["nutrition"]["calories"]
                
                return {
                    "food_name": item["title"],
                    "calories": calories,
                    "related": [x["title"] for x in menu_data["menuItems"][1:]]
                }
        
        return None
        
    except Exception as e:
        print(f"API Error: {e}")
        return None
