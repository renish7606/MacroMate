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
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Sum, Avg
from .models import UserProfile, FoodHistory
from datetime import timedelta


@login_required
def dashboard(request):
    """
    Dashboard UI with real data from profile and food history
    """
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    
    # Get user profile for calorie goal
    profile = None
    daily_goal = 2200  # Default fallback
    try:
        profile = request.user.profile
        if profile and profile.daily_calorie_goal:
            daily_goal = profile.daily_calorie_goal
    except UserProfile.DoesNotExist:
        pass
    
    # Get today's food entries
    today_foods = FoodHistory.objects.filter(
        user=request.user,
        created_at__date=today
    )
    
    # Calculate today's totals
    today_calories = today_foods.aggregate(total=Sum('calories'))['total'] or 0
    today_protein = today_foods.aggregate(total=Sum('protein'))['total'] or 0
    today_carbs = today_foods.aggregate(total=Sum('carbs'))['total'] or 0
    today_fat = today_foods.aggregate(total=Sum('fat'))['total'] or 0
    
    # Calculate remaining calories
    remaining_calories = max(0, daily_goal - today_calories)
    
    # Get weekly average (last 7 days)
    week_foods = FoodHistory.objects.filter(
        user=request.user,
        created_at__date__gte=week_ago,
        created_at__date__lte=today
    )
    week_avg_calories = week_foods.aggregate(avg=Avg('calories'))['avg'] or 0
    week_avg_calories = round(week_avg_calories * week_foods.count() / 7) if week_foods.count() > 0 else 0
    
    # Calculate macro percentages (rough estimates based on calorie goals)
    # Protein: ~25% of calories = ~4 cal/g, Carbs: ~45% = ~4 cal/g, Fat: ~30% = ~9 cal/g
    protein_target = round(daily_goal * 0.25 / 4)
    carbs_target = round(daily_goal * 0.45 / 4)
    fat_target = round(daily_goal * 0.30 / 9)
    
    protein_percent = min(100, round((today_protein / protein_target * 100))) if protein_target > 0 else 0
    carbs_percent = min(100, round((today_carbs / carbs_target * 100))) if carbs_target > 0 else 0
    fat_percent = min(100, round((today_fat / fat_target * 100))) if fat_target > 0 else 0
    
    # Status indicators
    remaining_status = "On track" if remaining_calories > daily_goal * 0.2 else "Low"
    protein_status = "Strong" if protein_percent >= 80 else "Balanced" if protein_percent >= 50 else "Low"
    carbs_status = "Balanced" if carbs_percent >= 50 else "Low"
    fat_status = "Stable" if fat_percent >= 50 else "Low"
    
    context = {
        'daily_goal': daily_goal,
        'today_calories': today_calories,
        'remaining_calories': remaining_calories,
        'week_avg_calories': week_avg_calories,
        'today_protein': round(today_protein, 1),
        'today_carbs': round(today_carbs, 1),
        'today_fat': round(today_fat, 1),
        'protein_target': protein_target,
        'carbs_target': carbs_target,
        'fat_target': fat_target,
        'protein_percent': protein_percent,
        'carbs_percent': carbs_percent,
        'fat_percent': fat_percent,
        'remaining_status': remaining_status,
        'protein_status': protein_status,
        'carbs_status': carbs_status,
        'fat_status': fat_status,
        'has_food_logged': today_foods.count() > 0,
        'profile': profile,
    }
    
    return render(request, "dashboard.html", context)


def upload_food(request):
    """
    Upload page: Image upload + food detection + confirmation flow
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
            "csv_foods": json.dumps(list(CSV_FOODS)),
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
        # Use higher threshold (80%) for truly confident predictions
        # Below 80%: ask for confirmation
        # Above 80%: show result even if nutrition not found
        HIGH_CONFIDENCE_THRESHOLD = 80.0
        
        if is_confident and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            # High confidence - don't ask for confirmation
            nutrition = get_nutrition(food)
            
            return render(request, "result.html", {
                "food": food.replace("_", " ").title(),
                "calories": nutrition["calories"] if nutrition else None,
                "protein": nutrition["protein"] if nutrition else None,
                "carbs": nutrition["carbs"] if nutrition else None,
                "fat": nutrition["fat"] if nutrition else None,
                "confidence": confidence,
                "raw_confidence": ml_result.get("raw_confidence", confidence),
                "needs_confirmation": False,
                "image": image_url,
                "csv_foods": json.dumps(list(CSV_FOODS)),
                "related": get_related_foods(food),
                "nutrition_available": nutrition is not None,
            })
        
        # -------- LOW CONFIDENCE PATH --------
        # Confidence below 80% - ask for confirmation
        print(f"⚠️ Low confidence ({confidence}%) - requesting manual confirmation")
        
        return render(request, "result.html", {
            "food": food.replace("_", " ").title(),  # Show best guess
            "calories": None,
            "confidence": confidence,
            "raw_confidence": ml_result.get("raw_confidence", confidence),
            "needs_confirmation": True,
            "image": image_url,
            "suggested_hint": food.replace("_", " "),
            "csv_foods": json.dumps(list(CSV_FOODS)),
            "related": ml_result.get("top_predictions", [])[:5],  # Show top 5 alternatives
            "nutrition_available": False,
        })

    # -----------------------------------------
    # 3️⃣ DEFAULT PAGE (GET REQUEST)
    # -----------------------------------------
    return render(request, "upload.html")


def analysis(request):
    """
    Detailed analysis UI (placeholders)
    """
    return render(request, "analysis.html")


def assistant(request):
    """
    Food-only assistant UI (demo guardrails)
    """
    return render(request, "assistant.html")


def _tdee_bmr(weight_kg, height_cm, age, gender, activity):
    """Mifflin–St Jeor BMR then TDEE."""
    if gender == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    factors = {"office": 1.2, "moderate": 1.55, "athlete": 1.725}
    return round(bmr * factors.get(activity, 1.2))


@login_required
def profile(request):
    """
    Profile: before filling → form with Name at top.
    After filling → show user's name, Edit button to open form.
    """
    edit_mode = request.GET.get("edit") == "1"
    profile_obj = None
    try:
        profile_obj = request.user.profile
    except UserProfile.DoesNotExist:
        pass

    def is_filled(p):
        return p and (p.name or (p.height and p.weight and p.age))

    has_filled = is_filled(profile_obj)
    show_form = not has_filled or edit_mode

    if request.method == "POST" and show_form:
        name = (request.POST.get("name") or "").strip()
        try:
            height = int(request.POST.get("height") or 0)
            weight = int(request.POST.get("weight") or 0)
            age = int(request.POST.get("age") or 0)
        except (TypeError, ValueError):
            height = weight = age = 0
        gender = request.POST.get("gender") or "male"
        activity = request.POST.get("activity") or "office"
        goal = request.POST.get("goal") or "maintain"

        if height and weight and age:
            daily = _tdee_bmr(weight, height, age, gender, activity)
        else:
            daily = 2200

        profile_obj, _ = UserProfile.objects.get_or_create(user=request.user)
        profile_obj.name = name or request.user.get_full_name() or request.user.username
        profile_obj.height = height or None
        profile_obj.weight = weight or None
        profile_obj.age = age or None
        profile_obj.gender = gender
        profile_obj.activity_level = activity
        profile_obj.goal = goal
        profile_obj.daily_calorie_goal = daily
        profile_obj.save()
        return redirect("profile")

    display_name = None
    if profile_obj and has_filled:
        display_name = profile_obj.name or request.user.get_full_name() or request.user.username

    context = {
        "profile": profile_obj,
        "has_filled": has_filled,
        "show_form": show_form,
        "display_name": display_name,
    }
    return render(request, "profile.html", context)


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
            food = request.POST.get("food", "").strip()
            calories = request.POST.get("calories", "").strip()
            protein = request.POST.get("protein", "0").strip()
            carbs = request.POST.get("carbs", "0").strip()
            fat = request.POST.get("fat", "0").strip()
            image = request.POST.get("image", "").strip()

            # Validate required fields
            if not food:
                return JsonResponse({"success": False, "error": "Food name is required"})
            
            # Check if calories is None or empty
            if not calories or calories.lower() == "none" or calories == "":
                return JsonResponse({"success": False, "error": "Calories information is required. Please enter a portion size first."})
            
            # Convert to numbers with proper error handling
            try:
                calories_int = int(float(calories))
                protein_float = float(protein) if protein and protein.lower() != "none" else 0.0
                carbs_float = float(carbs) if carbs and carbs.lower() != "none" else 0.0
                fat_float = float(fat) if fat and fat.lower() != "none" else 0.0
            except (ValueError, TypeError) as e:
                return JsonResponse({"success": False, "error": f"Invalid nutrition values: {str(e)}"})

            # Create the food history entry
            FoodHistory.objects.create(
                user=request.user,
                food=food,
                calories=calories_int,
                protein=protein_float,
                carbs=carbs_float,
                fat=fat_float,
                image=image
            )
            print(f"✅ Saved meal to history: {food} ({calories_int} kcal)")
            return JsonResponse({"success": True, "message": "Meal saved successfully!"})
                
        except Exception as e:
            print(f"❌ Error saving to history: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({"success": False, "error": f"Server error: {str(e)}"})

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
