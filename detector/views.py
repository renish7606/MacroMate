# import csv
# import os
# from django.conf import settings
# import json
# from django.shortcuts import render, redirect
# from services.nutrition_provider import get_nutrition
# from django.http import JsonResponse
# from services.portion_parser import parse_quantity
# from services.api_nutrition import get_nutrition_from_api
# from services.csv_nutrition import get_nutrition_from_csv
# from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import FileSystemStorage
# from datetime import datetime
# from .ml_food_predictor import predict_food
# from .food_api import fetch_food_info
# from .food_similarity import (
#     CSV_FOODS,
#     get_related_foods,
#     get_nutrition_for_food,
# )


# # --------------------------------------------------
# # MAIN PAGE + IMAGE UPLOAD
# # --------------------------------------------------
# # def index(request):
# #     """
# #     Image upload + food detection + confirmation flow
# #     """

# #     # CSV_FOODS = load_csv_foods()

# #     # Session data
# #     image = request.session.get("uploaded_image")
# #     confirmed_food = request.session.get("confirmed_food")

# #     # --------------------------------------------------
# #     # IF USER ALREADY CONFIRMED FOOD
# #     # --------------------------------------------------
# #     if confirmed_food:
# #         calories = get_calories_for_food(confirmed_food)
# #         related = get_related_foods(confirmed_food)

# #         return render(request, "result.html", {
# #             "food": confirmed_food.title(),
# #             "calories": calories,
# #             "related": related,
# #             "confidence": 100.0,
# #             "needs_confirmation": False,
# #             "csv_foods": list(CSV_FOODS),
# #             "image": image,
# #         })

# #     # --------------------------------------------------
# #     # IMAGE UPLOAD
# #     # --------------------------------------------------
# #     if request.method == "POST" and request.FILES.get("image"):
# #         image_file = request.FILES["image"]

# #         fs = FileSystemStorage()
# #         filename = fs.save(image_file.name, image_file)
# #         image_url = fs.url(filename)

# #         request.session["uploaded_image"] = image_url

# #         # --------------------------------------------------
# #         # FOOD API (fallback, NOT trusted blindly)
# #         # --------------------------------------------------
# #         api_result = fetch_food_info(image_file.name)

# #         if api_result:
# #             predicted_food = api_result.get("food_name", "").lower()
# #             confidence = 0.80
# #         else:
# #             predicted_food = "unknown food"
# #             confidence = 0.0

# #         # --------------------------------------------------
# #         # CSV FIRST LOGIC
# #         # --------------------------------------------------
# #         if predicted_food in CSV_FOODS and confidence >= 0.85:
# #             calories = get_calories_for_food(predicted_food)
# #             related = get_related_foods(predicted_food)

# #             return render(request, "result.html", {
# #                 "food": predicted_food.title(),
# #                 "calories": calories,
# #                 "related": related,
# #                 "confidence": round(confidence * 100, 2),
# #                 "needs_confirmation": False,
# #                 "csv_foods": list(CSV_FOODS),
# #                 "image": image_url,
# #             })

# #         # --------------------------------------------------
# #         # FORCE MANUAL CONFIRMATION
# #         # --------------------------------------------------
# #         return render(request, "result.html", {
# #             "food": predicted_food.title(),
# #             "calories": None,
# #             "related": [],
# #             "confidence": round(confidence * 100, 2),
# #             "needs_confirmation": True,
# #             "csv_foods": list(CSV_FOODS),
# #             "image": image_url,
# #         })

# #     # --------------------------------------------------
# #     # DEFAULT PAGE
# #     # --------------------------------------------------
# #     return render(request, "index.html")


# def index(request):

#     # -----------------------------------------
#     # 1️⃣ CONFIRMED FOOD (SESSION)
#     # -----------------------------------------
#     confirmed_food = request.session.get("confirmed_food")
#     image = request.session.get("uploaded_image")

#     if confirmed_food:
#         # nutrition = get_nutrition_for_food(confirmed_food)
#         nutrition = get_nutrition(confirmed_food)

#         return render(request, "result.html", {
#             "food": confirmed_food.replace("_", " ").title(),
#             "calories": nutrition["calories"] if nutrition else None,
#             "protein": nutrition["protein"] if nutrition else None,
#             "carbs": nutrition["carbs"] if nutrition else None,
#             "fat": nutrition["fat"] if nutrition else None,
#             "related": get_related_foods(confirmed_food),
#             "confidence": 100,
#             "needs_confirmation": False,
#             "image": image,
#             "csv_foods": list(CSV_FOODS),
#         })

#     # -----------------------------------------
#     # 2️⃣ IMAGE UPLOAD + ML
#     # -----------------------------------------
#     if request.method == "POST" and request.FILES.get("image"):
#         image_file = request.FILES["image"]

#         fs = FileSystemStorage()
#         filename = fs.save(image_file.name, image_file)
#         image_path = fs.path(filename)

#         request.session["uploaded_image"] = fs.url(filename)

#         ml_result = predict_food(image_path)
#         food = ml_result["food_name"]
#         confidence = ml_result["confidence"]

#         # -------- HIGH CONFIDENCE --------
#         if confidence >= 40:
#             # nutrition = get_nutrition_for_food(food)
#             nutrition = get_nutrition(confirmed_food)

#             return render(request, "result.html", {
#                 "food": food.replace("_", " ").title(),
#                 "calories": nutrition["calories"] if nutrition else None,
#                 "protein": nutrition["protein"] if nutrition else None,
#                 "carbs": nutrition["carbs"] if nutrition else None,
#                 "fat": nutrition["fat"] if nutrition else None,
#                 "confidence": confidence,
#                 "needs_confirmation": False,
#                 "image": request.session["uploaded_image"],
#                 "csv_foods": list(CSV_FOODS),
#             })

#         # -------- LOW CONFIDENCE --------
#         return render(request, "result.html", {
#             "food": "",
#             "calories": None,
#             "related": ml_result.get("top_predictions", []),
#             "confidence": confidence,
#             "needs_confirmation": True,
#             "image": request.session["uploaded_image"],
#             "suggested_hint": (
#                 ml_result["top_predictions"][0]["food"].replace("_", " ")
#                 if ml_result.get("top_predictions")
#                 else ""
#             ),
#             "csv_foods": list(CSV_FOODS),
#         })

#     return render(request, "index.html")



# # --------------------------------------------------
# # CONFIRM FOOD (USER INPUT)
# # --------------------------------------------------
# @csrf_exempt
# def confirm_food_and_log(request):
#     if request.method != "POST":
#         return JsonResponse({"success": False})

#     data = json.loads(request.body)
#     food = data.get("food", "").strip().lower().replace(" ", "_")

#     if not food:
#         return JsonResponse({"success": False})

#     request.session["confirmed_food"] = food
#     return JsonResponse({"success": True})


# # --------------------------------------------------
# # FOOD SEARCH (CSV)
# # --------------------------------------------------
# def get_food_suggestions_api(request):
#     q = request.GET.get("q", "").lower()
#     results = []

#     if len(q) >= 2:
#         for food in CSV_FOODS:
#             if q in food:
#                 results.append(food.replace("_", " ").title())

#     return JsonResponse({"suggestions": results[:10]})



# # --------------------------------------------------
# # MEAL HISTORY (PLACEHOLDER)
# # --------------------------------------------------
# from django.utils import timezone
# from django.contrib.auth.decorators import login_required
# from django.shortcuts import render
# from .models import FoodHistory

# @login_required
# def meal_history(request):
#     today = timezone.now().date()

#     meals = FoodHistory.objects.filter(
#         user=request.user,
#         created_at__date=today
#     ).order_by("-created_at")

#     total_calories = 0
#     total_protein = 0
#     total_carbs = 0
#     total_fat = 0

#     for meal in meals:
#         total_calories += meal.calories
#         total_protein += meal.protein
#         total_carbs += meal.carbs
#         total_fat += meal.fat

#     context = {
#         "today_date": today,
#         "meals": meals,
#         "meal_count": meals.count(),
#         "total_calories": total_calories,
#         "total_protein": total_protein,
#         "total_carbs": total_carbs,
#         "total_fat": total_fat,
#     }

#     return render(request, "meal_history.html", context)


# from django.http import JsonResponse
# from django.contrib.auth.decorators import login_required
# from .models import FoodHistory

# @login_required
# def save_to_history(request):
#     if request.method == "POST":
#         food = request.POST.get("food")
#         calories = request.POST.get("calories")
#         protein = request.POST.get("protein", 0)
#         carbs = request.POST.get("carbs", 0)
#         fat = request.POST.get("fat", 0)
#         image = request.POST.get("image", "")

#         if food and calories:
#             FoodHistory.objects.create(
#                 user=request.user,
#                 food=food,
#                 calories=int(calories),
#                 protein=float(protein),
#                 carbs=float(carbs),
#                 fat=float(fat),
#                 image=image
#             )
#             return JsonResponse({"success": True})

#     return JsonResponse({"success": False})


# from django.shortcuts import redirect

# def reset_analysis(request):
#     request.session.flush()
#     return redirect("index")

# from django.http import HttpResponse
# from django.contrib.auth.decorators import login_required
# from reportlab.lib.pagesizes import A4
# from reportlab.pdfgen import canvas
# from .models import FoodHistory

# @login_required
# def download_history_pdf(request):
#     response = HttpResponse(content_type='application/pdf')
#     response['Content-Disposition'] = 'attachment; filename="MacroMate_Report.pdf"'

#     p = canvas.Canvas(response, pagesize=A4)
#     width, height = A4

#     # Title
#     p.setFont("Helvetica-Bold", 16)
#     p.drawString(50, height - 50, "MacroMate - Food History Report")

#     y = height - 90
#     p.setFont("Helvetica", 11)

#     meals = FoodHistory.objects.filter(user=request.user).order_by('-created_at')
#     total_calories = 0

#     for meal in meals:
#         line = f"{meal.food} | {meal.calories} kcal | {meal.created_at.strftime('%d-%m-%Y %H:%M')}"
#         p.drawString(50, y, line)
#         y -= 18
#         total_calories += meal.calories

#         if y < 50:
#             p.showPage()
#             p.setFont("Helvetica", 11)
#             y = height - 50

#     # Total
#     p.setFont("Helvetica-Bold", 12)
#     p.drawString(50, y - 10, f"Total Calories: {total_calories} kcal")

#     p.showPage()
#     p.save()

#     return response


# # def analyze_food(request):
# #     if request.method == "POST":
# #         image = request.FILES.get("food_image")

# #         image_path = os.path.join("media/uploads", image.name)
# #         with open(image_path, "wb+") as f:
# #             for chunk in image.chunks():
# #                 f.write(chunk)

# #         ml_result = predict_food(image_path)

# #         if ml_result["is_confident"]:
# #             food_name = ml_result["food_name"]
# #             source = "ML Model"
# #         else:
# #             food_name = ml_result["food_name"]
# #             source = "Nutrition API"

# #         nutrition = get_food_nutrition(food_name)

# #         return render(request, "result.html", {
# #             "food_name": food_name,
# #             "confidence": ml_result["confidence"],
# #             "source": source,
# #             "calories": nutrition["calories"],
# #             "protein": nutrition["protein"],
# #             "carbs": nutrition["carbs"],
# #             "fat": nutrition["fat"],
# #             "suggestions": ml_result["top_predictions"]
# #         })


# # def load_calories():
# #     global CALORIE_MAP
# #     if CALORIE_MAP:
# #         return

# #     csv_path = os.path.join(settings.BASE_DIR, "data", "calories.csv")

# #     with open(csv_path, newline="", encoding="utf-8") as f:
# #         reader = csv.DictReader(f)

# #         for row in reader:
# #             # CSV header is: food , calorie
# #             food = row["food"].strip().lower().replace(" ", "_")
# #             calorie = row["calorie"].strip()

# #             CALORIE_MAP[food] = int(calorie)



# from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
# def parse_portion_api(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "Invalid method"}, status=400)

#     data = json.loads(request.body)
#     food = data.get("food")
#     portion_text = data.get("portion_text")

#     if not food or not portion_text:
#         return JsonResponse({"error": "Missing data"}, status=400)

#     # 1️⃣ Convert portion → grams
#     grams = parse_quantity(portion_text, food)

#     # 2️⃣ Try API nutrition
#     nutrition = get_nutrition_from_api(food , grams=grams)

#     # 3️⃣ Fallback to CSV
#     if not nutrition:
#         nutrition = get_nutrition_from_csv(food)

#     if not nutrition:
#         return JsonResponse({"error": "Nutrition not found"}, status=404)

#     # 4️⃣ Scale nutrition by grams (API is per default portion)
#     result = {
#     "grams": grams,
#     "calories": nutrition["calories"],
#     "protein": nutrition["protein"],
#     "carbs": nutrition["carbs"],
#     "fat": nutrition["fat"],
#     }


#     return JsonResponse(result)


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


def index(request):
    """
    Main page: Image upload + food detection + confirmation flow
    """
    
    # -----------------------------------------
    # 1️⃣ CONFIRMED FOOD (SESSION)
    # -----------------------------------------
    confirmed_food = request.session.get("confirmed_food")
    image = request.session.get("uploaded_image")

    if confirmed_food:
        # Get nutrition for confirmed food
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
    # 2️⃣ IMAGE UPLOAD + ML PREDICTION
    # -----------------------------------------
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Save uploaded image
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_path = fs.path(filename)
        image_url = fs.url(filename)

        request.session["uploaded_image"] = image_url

        # Run ML prediction
        ml_result = predict_food(image_path)
        
        # Extract results - FIX: Use correct variable names
        food = ml_result["food_name"]
        confidence = ml_result["confidence"]
        is_confident = ml_result["is_confident"]
        
        print(f"\n🎯 ML Result: {food} with confidence {confidence}%")
        print(f"   Is confident: {is_confident}")
        
        # -------- HIGH CONFIDENCE PATH --------
        if is_confident:
            # FIX: Use 'food' not 'confirmed_food' here!
            nutrition = get_nutrition(food)
            
            if nutrition:
                return render(request, "result.html", {
                    "food": food.replace("_", " ").title(),
                    "calories": nutrition["calories"],
                    "protein": nutrition["protein"],
                    "carbs": nutrition["carbs"],
                    "fat": nutrition["fat"],
                    "confidence": confidence,
                    "raw_confidence": ml_result.get("raw_confidence", confidence),
                    "needs_confirmation": False,
                    "image": image_url,
                    "csv_foods": list(CSV_FOODS),
                    "related": get_related_foods(food),
                })
            else:
                # Nutrition not found even for confident prediction
                print(f"⚠️ Warning: Nutrition not found for {food}")
                # Fall through to low confidence path
        
        # -------- LOW CONFIDENCE PATH --------
        print(f"⚠️ Low confidence ({confidence}%) - requesting manual confirmation")
        
        return render(request, "result.html", {
            "food": food.replace("_", " ").title(),  # Show best guess
            "calories": None,
            "confidence": confidence,
            "raw_confidence": ml_result.get("raw_confidence", confidence),
            "needs_confirmation": True,
            "image": image_url,
            "suggested_hint": food.replace("_", " "),
            "csv_foods": list(CSV_FOODS),
            "related": ml_result.get("top_predictions", [])[:5],  # Show top 5 alternatives
        })

    # -----------------------------------------
    # 3️⃣ DEFAULT PAGE (GET REQUEST)
    # -----------------------------------------
    return render(request, "index.html")


# --------------------------------------------------
# CONFIRM FOOD (USER INPUT)
# --------------------------------------------------
@csrf_exempt
def confirm_food_and_log(request):
    """
    Handle manual food confirmation from user
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Invalid method"})

    try:
        data = json.loads(request.body)
        food = data.get("food", "").strip().lower().replace(" ", "_")

        if not food:
            return JsonResponse({"success": False, "error": "No food specified"})

        # Save to session
        request.session["confirmed_food"] = food
        
        print(f"✅ User confirmed food: {food}")
        
        return JsonResponse({"success": True})
        
    except Exception as e:
        print(f"❌ Error in confirm_food_and_log: {e}")
        return JsonResponse({"success": False, "error": str(e)})


# --------------------------------------------------
# FOOD SEARCH (CSV AUTOCOMPLETE)
# --------------------------------------------------
def get_food_suggestions_api(request):
    """
    Return food suggestions from CSV based on search query
    """
    q = request.GET.get("q", "").lower()
    results = []

    if len(q) >= 2:
        for food in CSV_FOODS:
            if q in food:
                results.append(food.replace("_", " ").title())

    return JsonResponse({"suggestions": results[:10]})


# --------------------------------------------------
# MEAL HISTORY
# --------------------------------------------------
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from .models import FoodHistory

@login_required
def meal_history(request):
    """
    Display user's meal history for today
    """
    today = timezone.now().date()

    meals = FoodHistory.objects.filter(
        user=request.user,
        created_at__date=today
    ).order_by("-created_at")

    # Calculate totals
    total_calories = sum(meal.calories for meal in meals)
    total_protein = sum(meal.protein for meal in meals)
    total_carbs = sum(meal.carbs for meal in meals)
    total_fat = sum(meal.fat for meal in meals)

    context = {
        "today_date": today,
        "meals": meals,
        "meal_count": meals.count(),
        "total_calories": total_calories,
        "total_protein": round(total_protein, 1),
        "total_carbs": round(total_carbs, 1),
        "total_fat": round(total_fat, 1),
    }

    return render(request, "meal_history.html", context)


@login_required
def save_to_history(request):
    """
    Save meal to user's food history
    """
    if request.method == "POST":
        try:
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
                print(f"✅ Saved meal to history: {food}")
                return JsonResponse({"success": True})
            else:
                return JsonResponse({"success": False, "error": "Missing required fields"})
                
        except Exception as e:
            print(f"❌ Error saving to history: {e}")
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid method"})


def reset_analysis(request):
    """
    Clear session and start over
    """
    request.session.flush()
    return redirect("index")


from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

@login_required
def download_history_pdf(request):
    """
    Generate PDF report of user's food history
    """
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
    total_protein = 0
    total_carbs = 0
    total_fat = 0

    for meal in meals:
        line = f"{meal.food} | {meal.calories} kcal | P:{meal.protein}g C:{meal.carbs}g F:{meal.fat}g | {meal.created_at.strftime('%d-%m-%Y %H:%M')}"
        p.drawString(50, y, line)
        y -= 18
        
        total_calories += meal.calories
        total_protein += meal.protein
        total_carbs += meal.carbs
        total_fat += meal.fat

        if y < 50:
            p.showPage()
            p.setFont("Helvetica", 11)
            y = height - 50

    # Totals
    y -= 10
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, f"Total Calories: {total_calories} kcal")
    y -= 18
    p.drawString(50, y, f"Total Protein: {round(total_protein, 1)}g")
    y -= 18
    p.drawString(50, y, f"Total Carbs: {round(total_carbs, 1)}g")
    y -= 18
    p.drawString(50, y, f"Total Fat: {round(total_fat, 1)}g")

    p.showPage()
    p.save()

    return response


# --------------------------------------------------
# PORTION PARSING API
# --------------------------------------------------
@csrf_exempt
def parse_portion_api(request):
    """
    Parse user's portion description and calculate nutrition
    """
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)

    try:
        data = json.loads(request.body)
        food = data.get("food")
        portion_text = data.get("portion_text")

        if not food or not portion_text:
            return JsonResponse({"error": "Missing data"}, status=400)

        # 1️⃣ Convert portion → grams
        grams = parse_quantity(portion_text, food)
        print(f"📊 Parsed portion: '{portion_text}' = {grams}g of {food}")

        # 2️⃣ Try API nutrition first
        nutrition = get_nutrition_from_api(food, grams=grams)

        # 3️⃣ Fallback to CSV
        if not nutrition:
            print(f"⚠️ API failed, using CSV for {food}")
            base_nutrition = get_nutrition_from_csv(food)
            
            if base_nutrition:
                # Scale from 100g to actual grams
                scale_factor = grams / 100.0
                nutrition = {
                    "calories": int(base_nutrition["calories"] * scale_factor),
                    "protein": round(base_nutrition["protein"] * scale_factor, 1),
                    "carbs": round(base_nutrition["carbs"] * scale_factor, 1),
                    "fat": round(base_nutrition["fat"] * scale_factor, 1),
                }

        if not nutrition:
            return JsonResponse({"error": "Nutrition not found"}, status=404)

        # 4️⃣ Return calculated nutrition
        result = {
            "grams": grams,
            "calories": nutrition["calories"],
            "protein": nutrition["protein"],
            "carbs": nutrition["carbs"],
            "fat": nutrition["fat"],
        }

        print(f"✅ Returning nutrition: {result}")
        return JsonResponse(result)
        
    except Exception as e:
        print(f"❌ Error in parse_portion_api: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)
