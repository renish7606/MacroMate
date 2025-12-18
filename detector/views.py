
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from datetime import datetime


from .food_api import fetch_food_info
from .food_similarity import (
    CSV_FOODS,
    get_calories_for_food,
    get_related_foods,
)


# --------------------------------------------------
# MAIN PAGE + IMAGE UPLOAD
# --------------------------------------------------
def index(request):
    """
    Image upload + food detection + confirmation flow
    """

    # CSV_FOODS = load_csv_foods()

    # Session data
    image = request.session.get("uploaded_image")
    confirmed_food = request.session.get("confirmed_food")

    # --------------------------------------------------
    # IF USER ALREADY CONFIRMED FOOD
    # --------------------------------------------------
    if confirmed_food:
        calories = get_calories_for_food(confirmed_food)
        related = get_related_foods(confirmed_food)

        return render(request, "result.html", {
            "food": confirmed_food.title(),
            "calories": calories,
            "related": related,
            "confidence": 100.0,
            "needs_confirmation": False,
            "csv_foods": list(CSV_FOODS),
            "image": image,
        })

    # --------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_url = fs.url(filename)

        request.session["uploaded_image"] = image_url

        # --------------------------------------------------
        # FOOD API (fallback, NOT trusted blindly)
        # --------------------------------------------------
        api_result = fetch_food_info(image_file.name)

        if api_result:
            predicted_food = api_result.get("food_name", "").lower()
            confidence = 0.80
        else:
            predicted_food = "unknown food"
            confidence = 0.0

        # --------------------------------------------------
        # CSV FIRST LOGIC
        # --------------------------------------------------
        if predicted_food in CSV_FOODS and confidence >= 0.85:
            calories = get_calories_for_food(predicted_food)
            related = get_related_foods(predicted_food)

            return render(request, "result.html", {
                "food": predicted_food.title(),
                "calories": calories,
                "related": related,
                "confidence": round(confidence * 100, 2),
                "needs_confirmation": False,
                "csv_foods": list(CSV_FOODS),
                "image": image_url,
            })

        # --------------------------------------------------
        # FORCE MANUAL CONFIRMATION
        # --------------------------------------------------
        return render(request, "result.html", {
            "food": predicted_food.title(),
            "calories": None,
            "related": [],
            "confidence": round(confidence * 100, 2),
            "needs_confirmation": True,
            "csv_foods": list(CSV_FOODS),
            "image": image_url,
        })

    # --------------------------------------------------
    # DEFAULT PAGE
    # --------------------------------------------------
    return render(request, "index.html")


# --------------------------------------------------
# CONFIRM FOOD (USER INPUT)
# --------------------------------------------------
@csrf_exempt
def confirm_food_and_log(request):
    if request.method != "POST":
        return JsonResponse({"success": False})

    data = json.loads(request.body)
    food = data.get("food", "").strip().lower()

    if not food:
        return JsonResponse({"success": False})

    request.session["confirmed_food"] = food
    return JsonResponse({"success": True})


# --------------------------------------------------
# FOOD SEARCH (CSV)
# --------------------------------------------------
def get_food_suggestions_api(request):
    # CSV_FOODS = load_csv_foods()
    q = request.GET.get("q", "").lower()

    results = []
    if len(q) >= 2:
        for food in CSV_FOODS:
            if q in food:
                results.append(food.title())

    return JsonResponse({"suggestions": results[:10]})


# --------------------------------------------------
# MEAL HISTORY (PLACEHOLDER)
# --------------------------------------------------
from django.contrib.auth.decorators import login_required
from .models import FoodHistory
from datetime import date

@login_required
def meal_history(request):

    meals = FoodHistory.objects.filter(
        user=request.user
    ).order_by("-created_at")

    today = date.today()
    todays_meals = []
    total_calories = 0

    for meal in meals:
        calories = int(meal.calories)
        total_calories += calories

        todays_meals.append({
            "food_name": meal.food,
            "calories": calories,
            "get_time_display": meal.created_at.strftime("%H:%M"),
            "image": meal.image,
        })

    context = {
        "today_date": today,
        "todays_meals": todays_meals,
        "meal_count": len(todays_meals),
        "total_calories": total_calories,
    }

    return render(request, "meal_history.html", context)


from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import FoodHistory


@login_required
def save_to_history(request):
    if request.method != "POST":
        return JsonResponse({"success": False})

    food = request.POST.get("food")
    calories = request.POST.get("calories")
    image = request.POST.get("image")

    if not food or not calories:
        return JsonResponse({"success": False})

    FoodHistory.objects.create(
        user=request.user,      # 🔥 KEY FIX
        food=food,
        calories=calories,
        image=image
    )

    return JsonResponse({"success": True})


from django.shortcuts import redirect

def reset_analysis(request):
    return redirect("/")