import csv
import os
from django.conf import settings
import json
from django.shortcuts import render, redirect
from services.nutrition_provider import get_nutrition
from django.http import JsonResponse
from services.portion_parser import parse_quantity
from services.api_nutrition import get_nutrition_from_api
from services.csv_nutrition import get_nutrition_from_csv
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from .ml_food_predictor import predict_food
from .food_api import fetch_food_info
from .food_similarity import (
    CSV_FOODS,
    get_related_foods,
    get_nutrition_for_food,
)


# --------------------------------------------------
# MAIN PAGE + IMAGE UPLOAD
# --------------------------------------------------
# def index(request):
#     """
#     Image upload + food detection + confirmation flow
#     """

#     # CSV_FOODS = load_csv_foods()

#     # Session data
#     image = request.session.get("uploaded_image")
#     confirmed_food = request.session.get("confirmed_food")

#     # --------------------------------------------------
#     # IF USER ALREADY CONFIRMED FOOD
#     # --------------------------------------------------
#     if confirmed_food:
#         calories = get_calories_for_food(confirmed_food)
#         related = get_related_foods(confirmed_food)

#         return render(request, "result.html", {
#             "food": confirmed_food.title(),
#             "calories": calories,
#             "related": related,
#             "confidence": 100.0,
#             "needs_confirmation": False,
#             "csv_foods": list(CSV_FOODS),
#             "image": image,
#         })

#     # --------------------------------------------------
#     # IMAGE UPLOAD
#     # --------------------------------------------------
#     if request.method == "POST" and request.FILES.get("image"):
#         image_file = request.FILES["image"]

#         fs = FileSystemStorage()
#         filename = fs.save(image_file.name, image_file)
#         image_url = fs.url(filename)

#         request.session["uploaded_image"] = image_url

#         # --------------------------------------------------
#         # FOOD API (fallback, NOT trusted blindly)
#         # --------------------------------------------------
#         api_result = fetch_food_info(image_file.name)

#         if api_result:
#             predicted_food = api_result.get("food_name", "").lower()
#             confidence = 0.80
#         else:
#             predicted_food = "unknown food"
#             confidence = 0.0

#         # --------------------------------------------------
#         # CSV FIRST LOGIC
#         # --------------------------------------------------
#         if predicted_food in CSV_FOODS and confidence >= 0.85:
#             calories = get_calories_for_food(predicted_food)
#             related = get_related_foods(predicted_food)

#             return render(request, "result.html", {
#                 "food": predicted_food.title(),
#                 "calories": calories,
#                 "related": related,
#                 "confidence": round(confidence * 100, 2),
#                 "needs_confirmation": False,
#                 "csv_foods": list(CSV_FOODS),
#                 "image": image_url,
#             })

#         # --------------------------------------------------
#         # FORCE MANUAL CONFIRMATION
#         # --------------------------------------------------
#         return render(request, "result.html", {
#             "food": predicted_food.title(),
#             "calories": None,
#             "related": [],
#             "confidence": round(confidence * 100, 2),
#             "needs_confirmation": True,
#             "csv_foods": list(CSV_FOODS),
#             "image": image_url,
#         })

#     # --------------------------------------------------
#     # DEFAULT PAGE
#     # --------------------------------------------------
#     return render(request, "index.html")


def index(request):

    # -----------------------------------------
    # 1️⃣ CONFIRMED FOOD (SESSION)
    # -----------------------------------------
    confirmed_food = request.session.get("confirmed_food")
    image = request.session.get("uploaded_image")

    if confirmed_food:
        # nutrition = get_nutrition_for_food(confirmed_food)
        nutrition = get_nutrition(confirmed_food)

        return render(request, "result.html", {
            "food": confirmed_food.replace("_", " ").title(),
            "calories": nutrition["calories"] if nutrition else None,
            "protein": nutrition["protein"] if nutrition else None,
            "carbs": nutrition["carbs"] if nutrition else None,
            "fat": nutrition["fat"] if nutrition else None,
            "related": get_related_foods(confirmed_food),
            "confidence": 100,
            "needs_confirmation": False,
            "image": image,
            "csv_foods": list(CSV_FOODS),
        })

    # -----------------------------------------
    # 2️⃣ IMAGE UPLOAD + ML
    # -----------------------------------------
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_path = fs.path(filename)

        request.session["uploaded_image"] = fs.url(filename)

        ml_result = predict_food(image_path)
        food = ml_result["food_name"]
        confidence = ml_result["confidence"]

        # -------- HIGH CONFIDENCE --------
        if confidence >= 40:
            # nutrition = get_nutrition_for_food(food)
            nutrition = get_nutrition(confirmed_food)

            return render(request, "result.html", {
                "food": food.replace("_", " ").title(),
                "calories": nutrition["calories"] if nutrition else None,
                "protein": nutrition["protein"] if nutrition else None,
                "carbs": nutrition["carbs"] if nutrition else None,
                "fat": nutrition["fat"] if nutrition else None,
                "confidence": confidence,
                "needs_confirmation": False,
                "image": request.session["uploaded_image"],
                "csv_foods": list(CSV_FOODS),
            })

        # -------- LOW CONFIDENCE --------
        return render(request, "result.html", {
            "food": "",
            "calories": None,
            "related": ml_result.get("top_predictions", []),
            "confidence": confidence,
            "needs_confirmation": True,
            "image": request.session["uploaded_image"],
            "suggested_hint": (
                ml_result["top_predictions"][0]["food"].replace("_", " ")
                if ml_result.get("top_predictions")
                else ""
            ),
            "csv_foods": list(CSV_FOODS),
        })

    return render(request, "index.html")



# --------------------------------------------------
# CONFIRM FOOD (USER INPUT)
# --------------------------------------------------
@csrf_exempt
def confirm_food_and_log(request):
    if request.method != "POST":
        return JsonResponse({"success": False})

    data = json.loads(request.body)
    food = data.get("food", "").strip().lower().replace(" ", "_")

    if not food:
        return JsonResponse({"success": False})

    request.session["confirmed_food"] = food
    return JsonResponse({"success": True})


# --------------------------------------------------
# FOOD SEARCH (CSV)
# --------------------------------------------------
def get_food_suggestions_api(request):
    q = request.GET.get("q", "").lower()
    results = []

    if len(q) >= 2:
        for food in CSV_FOODS:
            if q in food:
                results.append(food.replace("_", " ").title())

    return JsonResponse({"suggestions": results[:10]})



# --------------------------------------------------
# MEAL HISTORY (PLACEHOLDER)
# --------------------------------------------------
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import FoodHistory

@login_required
def meal_history(request):
    today = timezone.now().date()

    meals = FoodHistory.objects.filter(
        user=request.user,
        created_at__date=today
    ).order_by("-created_at")

    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0

    for meal in meals:
        total_calories += meal.calories
        total_protein += meal.protein
        total_carbs += meal.carbs
        total_fat += meal.fat

    context = {
        "today_date": today,
        "meals": meals,
        "meal_count": meals.count(),
        "total_calories": total_calories,
        "total_protein": total_protein,
        "total_carbs": total_carbs,
        "total_fat": total_fat,
    }

    return render(request, "meal_history.html", context)


from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import FoodHistory

@login_required
def save_to_history(request):
    if request.method == "POST":
        food = request.POST.get("food")
        calories = request.POST.get("calories")
        protein = request.POST.get("protein", 0)
        carbs = request.POST.get("carbs", 0)
        fat = request.POST.get("fat", 0)
        image = request.POST.get("image", "")

        if food and calories:
            FoodHistory.objects.create(
                user=request.user,
                food=food,
                calories=int(calories),
                protein=float(protein),
                carbs=float(carbs),
                fat=float(fat),
                image=image
            )
            return JsonResponse({"success": True})

    return JsonResponse({"success": False})


from django.shortcuts import redirect

def reset_analysis(request):
    request.session.flush()
    return redirect("index")

from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .models import FoodHistory

@login_required
def download_history_pdf(request):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="MacroMate_Report.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "MacroMate - Food History Report")

    y = height - 90
    p.setFont("Helvetica", 11)

    meals = FoodHistory.objects.filter(user=request.user).order_by('-created_at')
    total_calories = 0

    for meal in meals:
        line = f"{meal.food} | {meal.calories} kcal | {meal.created_at.strftime('%d-%m-%Y %H:%M')}"
        p.drawString(50, y, line)
        y -= 18
        total_calories += meal.calories

        if y < 50:
            p.showPage()
            p.setFont("Helvetica", 11)
            y = height - 50

    # Total
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y - 10, f"Total Calories: {total_calories} kcal")

    p.showPage()
    p.save()

    return response


# def analyze_food(request):
#     if request.method == "POST":
#         image = request.FILES.get("food_image")

#         image_path = os.path.join("media/uploads", image.name)
#         with open(image_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         ml_result = predict_food(image_path)

#         if ml_result["is_confident"]:
#             food_name = ml_result["food_name"]
#             source = "ML Model"
#         else:
#             food_name = ml_result["food_name"]
#             source = "Nutrition API"

#         nutrition = get_food_nutrition(food_name)

#         return render(request, "result.html", {
#             "food_name": food_name,
#             "confidence": ml_result["confidence"],
#             "source": source,
#             "calories": nutrition["calories"],
#             "protein": nutrition["protein"],
#             "carbs": nutrition["carbs"],
#             "fat": nutrition["fat"],
#             "suggestions": ml_result["top_predictions"]
#         })


# def load_calories():
#     global CALORIE_MAP
#     if CALORIE_MAP:
#         return

#     csv_path = os.path.join(settings.BASE_DIR, "data", "calories.csv")

#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)

#         for row in reader:
#             # CSV header is: food , calorie
#             food = row["food"].strip().lower().replace(" ", "_")
#             calorie = row["calorie"].strip()

#             CALORIE_MAP[food] = int(calorie)



from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def parse_portion_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)

    data = json.loads(request.body)
    food = data.get("food")
    portion_text = data.get("portion_text")

    if not food or not portion_text:
        return JsonResponse({"error": "Missing data"}, status=400)

    # 1️⃣ Convert portion → grams
    grams = parse_quantity(portion_text, food)

    # 2️⃣ Try API nutrition
    nutrition = get_nutrition_from_api(food , grams=grams)

    # 3️⃣ Fallback to CSV
    if not nutrition:
        nutrition = get_nutrition_from_csv(food)

    if not nutrition:
        return JsonResponse({"error": "Nutrition not found"}, status=404)

    # 4️⃣ Scale nutrition by grams (API is per default portion)
    result = {
    "grams": grams,
    "calories": nutrition["calories"],
    "protein": nutrition["protein"],
    "carbs": nutrition["carbs"],
    "fat": nutrition["fat"],
    }


    return JsonResponse(result)
