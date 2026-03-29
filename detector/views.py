import csv
import os
import requests
from django.conf import settings
import json
from django.shortcuts import render, redirect
from services.nutrition_provider import get_nutrition
from django.http import JsonResponse
from services.portion_parser import parse_quantity
from services.api_nutrition import get_nutrition_from_api
from services.csv_nutrition import get_nutrition_from_csv
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import FileSystemStorage
from datetime import datetime
from .ml_food_predictor import predict_food, get_nutrition_from_model_db
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
import json as json_module


@login_required
def dashboard(request):
    """
    Dashboard UI with real data from profile and food history
    """
    print("[VIEW] dashboard requested")
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
    
    has_food_logged = today_foods.count() > 0

    # ── Micronutrient estimates (derived from today's food entries) ──
    sugar_g      = round(today_carbs * 0.10, 1)
    sugar_target = 25
    sugar_pct    = min(100, round(sugar_g / sugar_target * 100)) if sugar_target else 0

    fiber_g      = round(today_calories * 0.02, 1)
    fiber_target = 25
    fiber_pct    = min(100, round(fiber_g / fiber_target * 100)) if fiber_target else 0

    vitamin_a_pct  = min(100, round((today_calories / daily_goal) * 55)) if has_food_logged and daily_goal else 0
    vitamin_c_pct  = min(100, round((today_calories / daily_goal) * 40)) if has_food_logged and daily_goal else 0
    calcium_pct    = min(100, round((today_calories / daily_goal) * 35)) if has_food_logged and daily_goal else 0
    iron_pct       = min(100, round((today_calories / daily_goal) * 48)) if has_food_logged and daily_goal else 0

    # ── 7-day daily calorie data for chart ──
    chart_labels = []
    chart_calories_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        day_total = FoodHistory.objects.filter(
            user=request.user,
            created_at__date=day
        ).aggregate(total=Sum('calories'))['total'] or 0
        chart_labels.append(day.strftime('%a'))
        chart_calories_data.append(day_total)

    chart_macro_labels = ['Protein', 'Carbs', 'Fat']
    chart_macro_values = [round(today_protein, 1), round(today_carbs, 1), round(today_fat, 1)]

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
        'has_food_logged': has_food_logged,
        'profile': profile,
        # Micronutrients
        'sugar_g': sugar_g,
        'sugar_pct': sugar_pct,
        'fiber_g': fiber_g,
        'fiber_pct': fiber_pct,
        'vitamin_a_pct': vitamin_a_pct,
        'vitamin_c_pct': vitamin_c_pct,
        'calcium_pct': calcium_pct,
        'iron_pct': iron_pct,
        # Charts
        'chart_labels': json_module.dumps(chart_labels),
        'chart_calories': json_module.dumps(chart_calories_data),
        'chart_macro_labels': json_module.dumps(chart_macro_labels),
        'chart_macro_values': json_module.dumps(chart_macro_values),
        'daily_goal_line': daily_goal,
    }
    
    return render(request, "dashboard.html", context)


@login_required
def upload_food(request):
    """
    Upload page: handles 1 to 3 images.
    Single image → old result.html (backwards compatible).
    Multiple images → multi_result.html comparison.
    """

    print(f"[VIEW] upload_food {request.method}")

    # ── CONFIRMED FOOD (session, single-food legacy path) ──
    confirmed_food = request.session.get("confirmed_food")
    if confirmed_food:
        image     = request.session.get("uploaded_image")
        nutrition = get_nutrition(confirmed_food)
        return render(request, "result.html", {
            "food":               confirmed_food.replace("_", " ").title(),
            "calories":           nutrition["calories"] if nutrition else None,
            "protein":            nutrition["protein"]  if nutrition else None,
            "carbs":              nutrition["carbs"]    if nutrition else None,
            "fat":                nutrition["fat"]      if nutrition else None,
            "related":            get_related_foods(confirmed_food),
            "confidence":         100,
            "needs_confirmation": False,
            "image":              image,
            "csv_foods":          json.dumps(list(CSV_FOODS)),
        })

    # ── IMAGE UPLOAD ──
    if request.method == "POST" and request.FILES.getlist("image"):
        images = request.FILES.getlist("image")[:3]   # cap at 3
        fs     = FileSystemStorage()

        print(f"\n📸 Multi-image upload: received {len(images)} image(s)")

        results = []
        for idx, image_file in enumerate(images, 1):
            try:
                print(f"   Processing image {idx}/{len(images)}: {image_file.name}")
                filename   = fs.save(image_file.name, image_file)
                image_path = fs.path(filename)
                image_url  = fs.url(filename)

                ml_result  = predict_food(image_path)
                food_key   = ml_result["food_name"]
                confidence = ml_result["confidence"]
                method     = ml_result.get("method", "UNKNOWN")

                # Nutrition source routing:
                # CUSTOM -> class_names/label_nutrition_mapping
                # CLIP   -> Spoonacular API
                if method == "CUSTOM":
                    nutrition = get_nutrition_from_model_db(food_key) or {}
                elif method == "CLIP":
                    nutrition = get_nutrition_from_api(food_key.replace("_", " ")) or {}
                    # Keep the app usable if API quota/network fails.
                    if not nutrition:
                        nutrition = get_nutrition(food_key) or {}
                else:
                    nutrition = get_nutrition(food_key) or {}

                results.append({
                    "name":       food_key.replace("_", " ").title(),
                    "food_key":   food_key,
                    "image":      image_url,
                    "confidence": confidence,
                    "calories":   nutrition.get("calories", 0),
                    "protein":    nutrition.get("protein",  0),
                    "carbs":      nutrition.get("carbs",    0),
                    "fat":        nutrition.get("fat",      0),
                    "fiber":      nutrition.get("fiber",    0),
                    "sugar":      nutrition.get("sugar",    0),
                    "vitamin_a":  nutrition.get("vitamin_a", 0),
                    "vitamin_c":  nutrition.get("vitamin_c", 0),
                    "calcium":    nutrition.get("calcium",   0),
                    "iron":       nutrition.get("iron",      0),
                    "detection_method": method,
                })
            except Exception as e:
                print(f"   ⚠️ Error processing image {idx}: {e}")
                continue  # skip this image, process the rest

        if not results:
            return render(request, "upload.html", {
                "manual_error": "Could not process the uploaded images. Please try again.",
            })

        # ── SINGLE IMAGE → legacy result.html ──
        if len(results) == 1:
            r = results[0]
            method = r.get("detection_method", "UNKNOWN")
            # Custom model is your primary trained model, so allow lower
            # confidence than CLIP before asking manual confirmation.
            custom_confirm_threshold = 30.0
            clip_confirm_threshold = 65.0
            unknown_confirm_threshold = 80.0
            if method == "CUSTOM":
                confirm_threshold = custom_confirm_threshold
            elif method == "CLIP":
                confirm_threshold = clip_confirm_threshold
            else:
                confirm_threshold = unknown_confirm_threshold

            if r["confidence"] >= confirm_threshold:
                return render(request, "result.html", {
                    "food":               r["name"],
                    "calories":           r["calories"],
                    "protein":            r["protein"],
                    "carbs":              r["carbs"],
                    "fat":                r["fat"],
                    "fiber":              r.get("fiber", 0),
                    "sugar":              r.get("sugar", 0),
                    "vitamin_a":          r.get("vitamin_a", 0),
                    "vitamin_c":          r.get("vitamin_c", 0),
                    "calcium":            r.get("calcium", 0),
                    "iron":               r.get("iron", 0),
                    "detection_method":   r.get("detection_method", "UNKNOWN"),
                    "confidence":         r["confidence"],
                    "needs_confirmation": False,
                    "image":              r["image"],
                    "csv_foods":          json.dumps(list(CSV_FOODS)),
                    "related":            get_related_foods(r["food_key"]),
                    "nutrition_available": True,
                })
            else:
                return render(request, "result.html", {
                    "food":               r["name"],
                    "calories":           None,
                    "confidence":         r["confidence"],
                    "needs_confirmation": True,
                    "image":              r["image"],
                    "suggested_hint":     r["food_key"].replace("_", " "),
                    "csv_foods":          json.dumps(list(CSV_FOODS)),
                    "related":            [],
                    "nutrition_available": False,
                })

        # ── MULTIPLE IMAGES → multi_result.html ──
        return _render_multi_result(request, results)

    # ── DEFAULT GET ──
    return render(request, "upload.html")


def _render_multi_result(request, results):
    """
    Shared helper: annotates results with best-choice logic and renders multi_result.html
    """
    if not results:
        return redirect("upload_food")

    # Best choice = highest protein-to-calorie ratio (or lowest calories as tiebreaker)
    def score(r):
        cal = r["calories"] or 1
        return (r["protein"] / cal * 100) - (cal / 500)

    best_idx = max(range(len(results)), key=lambda i: score(results[i]))

    # Max calories for the progress bar width
    max_cal = max((r["calories"] for r in results), default=1) or 1

    for i, r in enumerate(results):
        r["is_best"]     = (i == best_idx)
        r["calorie_pct"] = round(r["calories"] / max_cal * 100)

    best      = results[best_idx]
    names     = [r["name"] for r in results]
    best_reason = f"lower calories ({best['calories']} kcal) and higher protein ({best['protein']}g)"

    # Display string: "Pizza, Samosa and Ramen"
    if len(names) == 1:
        food_names_display = names[0]
    elif len(names) == 2:
        food_names_display = f"{names[0]} and {names[1]}"
    else:
        food_names_display = f"{names[0]}, {names[1]} and {names[2]}"

    return render(request, "multi_result.html", {
        "foods":              results,
        "best_food":          best,
        "best_reason":        best_reason,
        "food_names_display": food_names_display,
    })


@login_required
def analysis(request):
    """
    Detailed analysis with real aggregated data and Chart.js charts
    """
    print("[VIEW] analysis requested")
    today = timezone.now().date()
    week_ago = today - timedelta(days=6)

    entries = FoodHistory.objects.filter(
        user=request.user,
        created_at__date__gte=week_ago,
        created_at__date__lte=today
    ).order_by('created_at')

    daily_data = {}
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        daily_data[d] = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}

    for e in entries:
        d = timezone.localdate(e.created_at)
        if d in daily_data:
            daily_data[d]['calories'] += e.calories
            daily_data[d]['protein']  += e.protein
            daily_data[d]['carbs']    += e.carbs
            daily_data[d]['fat']      += e.fat

    chart_labels   = [d.strftime('%a') for d in daily_data]
    chart_calories = [daily_data[d]['calories'] for d in daily_data]
    chart_protein  = [round(daily_data[d]['protein'], 1) for d in daily_data]
    chart_carbs    = [round(daily_data[d]['carbs'], 1)   for d in daily_data]
    chart_fat      = [round(daily_data[d]['fat'], 1)     for d in daily_data]
    chart_sugar    = [round(c * 0.10, 1) for c in chart_carbs]

    total_cal    = sum(chart_calories)
    total_prot   = sum(chart_protein)
    total_carb_w = sum(chart_carbs)
    total_fat_w  = sum(chart_fat)

    daily_goal = 2200
    try:
        daily_goal = request.user.profile.daily_calorie_goal or 2200
    except Exception:
        pass

    total_macro_cal = total_prot * 4 + total_carb_w * 4 + total_fat_w * 9
    if total_macro_cal > 0:
        prot_pct = round(total_prot * 4 / total_macro_cal * 100)
        carb_pct = round(total_carb_w * 4 / total_macro_cal * 100)
        fat_pct  = round(total_fat_w * 9 / total_macro_cal * 100)
    else:
        prot_pct = carb_pct = fat_pct = 0

    macro_status = 'Balanced' if 20 <= prot_pct <= 35 and 35 <= carb_pct <= 55 else 'Unbalanced'

    avg_daily_sugar = round(sum(chart_sugar) / 7, 1)
    sugar_status = 'Warning' if avg_daily_sugar > 20 else 'Good'

    avg_daily_cal = total_cal / 7 if total_cal else 0
    vitamin_c_pct = min(100, round(avg_daily_cal / daily_goal * 40)) if daily_goal else 0
    fiber_pct     = min(100, round(avg_daily_cal / daily_goal * 46)) if daily_goal else 0
    sodium_status = 'High' if total_fat_w > 100 else 'Normal'

    context = {
        'chart_labels':       json_module.dumps(chart_labels),
        'chart_calories':     json_module.dumps(chart_calories),
        'chart_protein':      json_module.dumps(chart_protein),
        'chart_carbs':        json_module.dumps(chart_carbs),
        'chart_fat':          json_module.dumps(chart_fat),
        'chart_sugar':        json_module.dumps(chart_sugar),
        'daily_goal':         daily_goal,
        'macro_status':       macro_status,
        'prot_pct':           prot_pct,
        'carb_pct':           carb_pct,
        'fat_pct':            fat_pct,
        'sugar_status':       sugar_status,
        'avg_daily_sugar':    avg_daily_sugar,
        'vitamin_c_pct':      vitamin_c_pct,
        'fiber_pct':          fiber_pct,
        'sodium_status':      sodium_status,
        'total_cal':          total_cal,
        'has_data':           total_cal > 0,
        # Pre-computed tips for template (avoids < in templates)
        'vitamin_c_tip':      'Likely low \u2022 add citrus/berries' if vitamin_c_pct < 50 else 'On track',
        'fiber_tip':          'Moderate \u2022 add legumes/oats' if fiber_pct < 50 else 'Good intake',
        'sodium_tip':         'Potentially high \u2022 monitor snacks' if sodium_status == 'High' else 'Normal levels',
        'sodium_bar':         72 if sodium_status == 'High' else 38,
    }
    return render(request, 'analysis.html', context)


@login_required
def assistant(request):
    """
    Food-only assistant UI
    """
    print("[VIEW] assistant page requested")
    return render(request, "assistant.html")


# ── Food keyword guardrail ──
FOOD_KEYWORDS = [
    'food', 'nutrition', 'diet', 'calorie', 'calories', 'macro', 'macros',
    'protein', 'carb', 'carbs', 'fat', 'fiber', 'vitamin', 'minerals',
    'health', 'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'weight',
    'sugar', 'salt', 'sodium', 'cholesterol', 'hydration', 'water',
    'fruit', 'fruits', 'vegetable', 'vegetables', 'recipe', 'cook',
    'eat', 'eating', 'hungry', 'rice', 'bread', 'roti', 'samosa',
    'biryani', 'dal', 'oats', 'egg', 'chicken', 'fish', 'milk',
    'cheese', 'yogurt', 'juice', 'tea', 'coffee', 'oil', 'butter',
    'pizza', 'burger', 'salad', 'soup', 'pasta', 'noodle', 'paneer',
    'dosa', 'idli', 'vada', 'pakoda', 'keto', 'vegan', 'gluten',
    'iron', 'calcium', 'potassium', 'magnesium', 'zinc', 'omega',
    'kcal', 'bmi', 'tdee', 'serving', 'portion', 'gram', 'grams',
]


def _looks_food_related(text):
    """Check if the question is likely about food/nutrition."""
    t = (text or '').lower()
    return any(kw in t for kw in FOOD_KEYWORDS)


@login_required
@require_POST
@csrf_exempt
def assistant_chat(request):
    """
    Real assistant endpoint — uses CSV nutrition DB + Spoonacular API.
    Three-tier strategy:
      1. Spoonacular Quick Answer (factual nutrition Qs)
      2. Local CSV/API nutrition lookup (extract food name from question)
      3. Spoonacular Chatbot (conversational fallback)
    """
    print("[VIEW] assistant_chat POST received")
    try:
        data = json.loads(request.body)
        question = (data.get("message") or "").strip()

        if not question:
            return JsonResponse({"reply": "Please type a question."})

        # ── Guardrail: reject obviously non-food questions ──
        if not _looks_food_related(question):
            return JsonResponse({
                "reply": "I'm MacroMate — I only help with food and nutrition questions. "
                         "Try asking about calories, macros, or healthy meal choices!",
                "is_refusal": True,
            })

        api_key = settings.SPOONACULAR_API_KEY
        if not api_key:
            return JsonResponse({"reply": "Assistant is not configured. Add SPOONACULAR_API_KEY to .env."})

        reply = None

        # ── Strategy 1: Quick Answer (best for factual nutrition Qs) ──
        try:
            qa_resp = requests.get(
                "https://api.spoonacular.com/recipes/quickAnswer",
                params={"q": question, "apiKey": api_key},
                timeout=10,
            )
            if qa_resp.status_code == 200:
                qa_data = qa_resp.json()
                answer = qa_data.get("answer", "")
                # Only accept if it's a real answer, not a generic redirect
                if answer and "here are some" not in answer.lower():
                    reply = answer
        except Exception:
            pass

        # ── Strategy 2: Local CSV/API nutrition lookup ──
        if not reply:
            reply = _try_local_nutrition_lookup(question)

        # ── Strategy 3: Chatbot (conversational fallback) ──
        if not reply:
            try:
                context_id = request.session.get("spoonacular_context_id", "")
                chat_params = {
                    "text": question,
                    "apiKey": api_key,
                }
                if context_id:
                    chat_params["contextId"] = context_id

                chat_resp = requests.get(
                    "https://api.spoonacular.com/food/converse",
                    params=chat_params,
                    timeout=10,
                )
                if chat_resp.status_code == 200:
                    chat_data = chat_resp.json()
                    answer = chat_data.get("answerText", "")
                    # Only accept if it's a real answer
                    if answer and "here are some" not in answer.lower():
                        reply = answer
                    # Save context for follow-up questions
                    if chat_data.get("contextId"):
                        request.session["spoonacular_context_id"] = chat_data["contextId"]
                        request.session.modified = True
            except Exception:
                pass

        if not reply:
            reply = "Sorry, I couldn't find an answer. Try rephrasing your food or nutrition question."

        return JsonResponse({"reply": reply})

    except Exception as e:
        print(f"Assistant error: {e}")
        return JsonResponse({"reply": "Something went wrong. Please try again."})


def _try_local_nutrition_lookup(question):
    """
    Try to extract a food name from the user's question and look it up
    in the local CSV nutrition database.
    Returns a formatted answer string or None.
    """
    import re

    q = question.lower().strip()

    # Strip common question patterns to isolate the food name
    # e.g. "how many calories in 2 samosa" → "samosa"
    #      "calorie in 1 samosa"            → "samosa"
    #      "nutrition of roti"              → "roti"
    patterns = [
        r'(?:how\s+(?:many|much)\s+)?(?:calories?|calorie|kcal|protein|carbs?|fat|nutrition|macro|macros|fiber|sugar|vitamin\w*)\s+(?:in|of|for)\s+(?:\d+\s*)?',
        r'(?:what\s+(?:is|are)\s+(?:the\s+)?)?(?:calories?|calorie|kcal|protein|carbs?|fat|nutrition|macro|macros)\s+(?:in|of|for)\s+(?:\d+\s*)?',
        r'(?:tell\s+me\s+(?:about|the)\s+)?(?:nutrition|calories?|macros?)\s+(?:of|in|for)\s+(?:\d+\s*)?',
        r'(?:how\s+(?:many|much)\s+)?(?:calories?|calorie|kcal|protein|carbs?|fat)\s+(?:does|do)\s+(?:\d+\s*)?',
    ]

    food_name = None
    for pat in patterns:
        match = re.search(pat, q)
        if match:
            food_name = q[match.end():].strip()
            break

    # Also try simple patterns: "samosa calories", "about samosa"
    if not food_name:
        simple = re.sub(
            r'\b(how|many|much|what|is|are|the|tell|me|about|does|do|have|has|'
            r'calories?|calorie|kcal|protein|carbs?|fat|nutrition|macro|macros|'
            r'fiber|sugar|vitamin\w*|in|of|for|per|serving|piece|pieces|'
            r'a|an|one|two|three|1|2|3|4|5|6|7|8|9|0)\b',
            ' ', q
        )
        food_name = ' '.join(simple.split()).strip()

    if not food_name or len(food_name) < 2:
        return None

    # Clean up: remove trailing punctuation, question marks
    food_name = re.sub(r'[?.!,]+$', '', food_name).strip()

    # Try exact match first, then partial match
    nutrition = get_nutrition(food_name)

    if not nutrition:
        # Try with underscores (CSV format)
        nutrition = get_nutrition(food_name.replace(' ', '_'))

    if not nutrition:
        # Try matching against known CSV foods
        from .food_similarity import CSV_FOODS
        for csv_food in CSV_FOODS:
            csv_clean = csv_food.replace('_', ' ')
            if csv_clean in food_name or food_name in csv_clean:
                nutrition = get_nutrition(csv_food)
                if nutrition:
                    food_name = csv_clean
                    break

    if nutrition:
        display = food_name.replace('_', ' ').title()
        cal = nutrition['calories']
        prot = nutrition['protein']
        carbs = nutrition['carbs']
        fat = nutrition['fat']

        return (
            f"{display} (per 100g serving):\n"
            f"• Calories: {cal} kcal\n"
            f"• Protein: {prot}g\n"
            f"• Carbs: {carbs}g\n"
            f"• Fat: {fat}g"
        )

    return None


@login_required
def clear_assistant_history(request):
    """
    Clear chatbot context for this user's session.
    """
    if "spoonacular_context_id" in request.session:
        del request.session["spoonacular_context_id"]
        request.session.modified = True
    return JsonResponse({"success": True})


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


@login_required
def reset_analysis(request):
    """
    Clear only food-related session keys.
    Never touch auth session keys — that would log the user out.
    """
    FOOD_SESSION_KEYS = [
        "confirmed_food",
        "uploaded_image",
        "assistant_history",
        "manual_query",
    ]
    for key in FOOD_SESSION_KEYS:
        request.session.pop(key, None)   # safe delete — ignores missing keys

    request.session.modified = True
    return redirect("upload_food")


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


@login_required
def download_today_pdf(request):
    """
    Generate PDF of today's food log only
    """
    from reportlab.lib import colors

    response = HttpResponse(content_type='application/pdf')
    today = timezone.now().date()
    response['Content-Disposition'] = f'attachment; filename="MacroMate_{today}.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # Header bar
    p.setFillColor(colors.HexColor('#a3e635'))
    p.rect(0, height - 60, width, 60, fill=True, stroke=False)
    p.setFillColor(colors.HexColor('#0a0a0f'))
    p.setFont("Helvetica-Bold", 18)
    p.drawString(40, height - 40, "MacroMate — Today's Food Log")

    p.setFillColor(colors.black)
    p.setFont("Helvetica", 11)
    p.drawString(40, height - 80, f"Date: {today.strftime('%B %d, %Y')}")
    p.drawString(40, height - 96, f"User: {request.user.username}")

    meals = FoodHistory.objects.filter(
        user=request.user,
        created_at__date=today
    ).order_by('created_at')

    y = height - 130
    total_cal = total_p = total_c = total_f = 0

    # Table header
    p.setFillColor(colors.HexColor('#f0f0f5'))
    p.rect(40, y - 4, width - 80, 20, fill=True, stroke=False)
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 10)
    p.drawString(45, y + 2, "Food")
    p.drawString(240, y + 2, "Time")
    p.drawString(310, y + 2, "Calories")
    p.drawString(390, y + 2, "Protein")
    p.drawString(450, y + 2, "Carbs")
    p.drawString(510, y + 2, "Fat")
    y -= 24

    p.setFont("Helvetica", 10)
    for meal in meals:
        t = timezone.localtime(meal.created_at).strftime('%H:%M')
        p.drawString(45,  y, meal.food[:28])
        p.drawString(240, y, t)
        p.drawString(310, y, f"{meal.calories} kcal")
        p.drawString(390, y, f"{meal.protein}g")
        p.drawString(450, y, f"{meal.carbs}g")
        p.drawString(510, y, f"{meal.fat}g")
        total_cal += meal.calories
        total_p   += meal.protein
        total_c   += meal.carbs
        total_f   += meal.fat
        y -= 18
        if y < 80:
            p.showPage()
            p.setFont("Helvetica", 10)
            y = height - 50

    # Totals row
    y -= 8
    p.setFillColor(colors.HexColor('#f0f0f5'))
    p.rect(40, y - 4, width - 80, 20, fill=True, stroke=False)
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 10)
    p.drawString(45,  y + 2, "TOTAL")
    p.drawString(310, y + 2, f"{total_cal} kcal")
    p.drawString(390, y + 2, f"{round(total_p,1)}g")
    p.drawString(450, y + 2, f"{round(total_c,1)}g")
    p.drawString(510, y + 2, f"{round(total_f,1)}g")

    p.showPage()
    p.save()
    return response


@login_required
def manual_food_lookup(request):
    """
    Handle 1-3 manual food name submissions.
    Single food → result.html.
    Multiple foods → multi_result.html comparison.
    """
    if request.method != "POST":
        return redirect("upload_food")

    # Collect up to 3 food name fields: manual_food_1, manual_food_2, manual_food_3
    raw_names = [
        request.POST.get("manual_food_1", "").strip(),
        request.POST.get("manual_food_2", "").strip(),
        request.POST.get("manual_food_3", "").strip(),
    ]
    # Remove blanks and normalize
    food_keys = [n.lower().replace(" ", "_") for n in raw_names if n]

    if not food_keys:
        return redirect("upload_food")

    results = []
    not_found = []

    for key in food_keys:
        nutrition = get_nutrition(key)
        if not nutrition:
            # Try fuzzy match from CSV_FOODS
            from difflib import get_close_matches
            matches = get_close_matches(key, CSV_FOODS, n=1, cutoff=0.6)
            if matches:
                key       = matches[0]
                nutrition = get_nutrition(key)

        if nutrition:
            results.append({
                "name":       key.replace("_", " ").title(),
                "food_key":   key,
                "image":      None,
                "confidence": None,
                "calories":   nutrition["calories"],
                "protein":    nutrition["protein"],
                "carbs":      nutrition["carbs"],
                "fat":        nutrition["fat"],
            })
        else:
            not_found.append(key.replace("_", " ").title())

    if not results:
        return render(request, "upload.html", {
            "manual_error": f"None of the foods were found: {', '.join(not_found)}. Try different names.",
        })

    if len(results) == 1:
        r = results[0]
        return render(request, "result.html", {
            "food":               r["name"],
            "calories":           r["calories"],
            "protein":            r["protein"],
            "carbs":              r["carbs"],
            "fat":                r["fat"],
            "confidence":         100,
            "needs_confirmation": False,
            "image":              None,
            "csv_foods":          json.dumps(list(CSV_FOODS)),
            "related":            get_related_foods(r["food_key"]),
            "nutrition_available": True,
        })

    return _render_multi_result(request, results)


# --------------------------------------------------
# PORTION PARSING API
# --------------------------------------------------
@csrf_exempt
def parse_portion_api(request):
    """
    Parse user's portion description and calculate nutrition
    """
    print("[VIEW] parse_portion_api called")
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
