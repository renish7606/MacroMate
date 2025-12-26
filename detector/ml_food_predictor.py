# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import os

# # ================= CONFIG =================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL_PATH = os.path.join(BASE_DIR, "models", "food101_efficientnet.keras")
# LABELS_PATH = os.path.join(BASE_DIR, "data", "food101_labels.txt")

# IMG_SIZE = 224
# CONFIDENCE_THRESHOLD = 40
# # ==========================================

# # Load model once
# model = load_model(MODEL_PATH)

# # Load labels
# with open(LABELS_PATH, "r") as f:
#     CLASS_LABELS = [line.strip() for line in f.readlines()]


# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("RGB")
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     img = np.array(img) / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img


# # def predict_food(image_path):
# #     img = preprocess_image(image_path)
    
# #     logits = model.predict(img)[0]
# #     predictions = tf.nn.softmax(logits).numpy()

# #     best_index = int(np.argmax(predictions))
# #     confidence = float(predictions[best_index])

# #     return {
# #         "food_name": CLASS_LABELS[best_index],
# #         "confidence": round(confidence, 2),
# #         "is_confident": confidence >= CONFIDENCE_THRESHOLD,
# #         "top_predictions": get_top_predictions(predictions)
# #     }

# def predict_food(image_path):
#     img = preprocess_image(image_path)

#     # Model output (logits)
#     logits = model.predict(img)[0]

#     # Convert logits → probabilities
#     # predictions = tf.nn.softmax(logits).numpy()
#     predictions = softmax_with_temperature(logits, temperature=1.5)

#     best_index = int(np.argmax(predictions))
#     confidence = float(predictions[best_index])

#     return {
#         "food_name": CLASS_LABELS[best_index],
#         "confidence": round(confidence * 100, 2),  # %
#         "is_confident": confidence >= 0.40,
#         "top_predictions": [
#             {
#                 "food": CLASS_LABELS[i],
#                 "confidence": round(float(predictions[i]) * 100, 2)
#             }
#             for i in np.argsort(predictions)[-5:][::-1]
#         ]
#     }


# def get_top_predictions(predictions, top_k=5):
#     indices = np.argsort(predictions)[-top_k:][::-1]
#     return [
#         {
#             "food": CLASS_LABELS[i],
#             "confidence": round(float(predictions[i])*100, 2)
#         }
#         for i in indices
        
#     ]

# def softmax_with_temperature(logits, temperature=1.5):
#     exp = np.exp(logits / temperature)
#     return exp / np.sum(exp)


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageEnhance
# import os

# # ================= CONFIG =================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODEL_PATH = os.path.join(BASE_DIR, "models", "food101_efficientnet.keras")
# LABELS_PATH = os.path.join(BASE_DIR, "data", "food101_labels.txt")

# IMG_SIZE = 224
# CONFIDENCE_THRESHOLD = 0.25  # 25% - will ask for confirmation below this
# # ==========================================

# # Load model
# try:
#     model = load_model(MODEL_PATH)
#     print(f"✅ Model loaded successfully")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# # Load labels
# try:
#     with open(LABELS_PATH, "r") as f:
#         CLASS_LABELS = [line.strip() for line in f.readlines()]
#     print(f"✅ Loaded {len(CLASS_LABELS)} food labels")
# except Exception as e:
#     print(f"❌ Error loading labels: {e}")
#     CLASS_LABELS = []


# def preprocess_image_variant(image_path, brightness=1.0, contrast=1.0, rotation=0):
#     """
#     Preprocess with augmentation variants for ensemble prediction
#     """
#     try:
#         img = Image.open(image_path).convert("RGB")
        
#         # Apply augmentations
#         if brightness != 1.0:
#             enhancer = ImageEnhance.Brightness(img)
#             img = enhancer.enhance(brightness)
        
#         if contrast != 1.0:
#             enhancer = ImageEnhance.Contrast(img)
#             img = enhancer.enhance(contrast)
        
#         if rotation != 0:
#             img = img.rotate(rotation, expand=False)
        
#         # Resize and normalize
#         img = img.resize((IMG_SIZE, IMG_SIZE))
#         img_array = np.array(img, dtype=np.float32)
#         img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)
        
#         return img_array
#     except Exception as e:
#         print(f"❌ Error preprocessing: {e}")
#         return None


# def predict_with_tta(image_path, model):
#     """
#     Test-Time Augmentation: Predict on multiple variants and average
#     This improves accuracy by considering different views of the image
#     """
#     variants = [
#         {"brightness": 1.0, "contrast": 1.0, "rotation": 0},    # Original
#         {"brightness": 1.1, "contrast": 1.0, "rotation": 0},    # Brighter
#         {"brightness": 0.9, "contrast": 1.0, "rotation": 0},    # Darker
#         {"brightness": 1.0, "contrast": 1.1, "rotation": 0},    # More contrast
#         {"brightness": 1.0, "contrast": 0.9, "rotation": 0},    # Less contrast
#     ]
    
#     all_predictions = []
    
#     for variant in variants:
#         img = preprocess_image_variant(
#             image_path,
#             brightness=variant["brightness"],
#             contrast=variant["contrast"],
#             rotation=variant["rotation"]
#         )
        
#         if img is not None:
#             pred = model.predict(img, verbose=0)[0]
#             all_predictions.append(pred)
    
#     if not all_predictions:
#         return None
    
#     # Average predictions across all variants
#     averaged_predictions = np.mean(all_predictions, axis=0)
    
#     # Normalize if needed
#     if not (0.99 < averaged_predictions.sum() < 1.01):
#         averaged_predictions = tf.nn.softmax(averaged_predictions).numpy()
    
#     return averaged_predictions


# def calculate_entropy(probabilities):
#     """Calculate entropy to measure uncertainty"""
#     probabilities = np.clip(probabilities, 1e-10, 1.0)
#     entropy = -np.sum(probabilities * np.log(probabilities))
#     return entropy


# def get_calibrated_confidence(probabilities):
#     """
#     Enhanced calibration considering multiple factors
#     """
#     sorted_probs = np.sort(probabilities)[::-1]
    
#     # Top predictions
#     top1 = sorted_probs[0]
#     top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0
#     top3 = sorted_probs[2] if len(sorted_probs) > 2 else 0
    
#     # Gaps
#     gap_1_2 = top1 - top2
#     gap_2_3 = top2 - top3
    
#     # Entropy
#     entropy = calculate_entropy(probabilities)
#     max_entropy = np.log(len(probabilities))
#     normalized_entropy = entropy / max_entropy
    
#     # Enhanced calibration formula
#     gap_factor = 1 + gap_1_2 * (1 + gap_2_3)
#     entropy_factor = (1 - normalized_entropy * 0.6)
    
#     confidence = top1 * gap_factor * entropy_factor
    
#     return np.clip(confidence, 0.0, 1.0)


# def check_csv_availability(food_name):
#     """
#     Check if food exists in CSV nutrition database
#     Boost confidence if we have nutrition data for it
#     """
#     try:
#         import csv
#         from django.conf import settings
        
#         csv_path = os.path.join(settings.BASE_DIR, "data", "calories.csv")
        
#         food_key = food_name.lower().replace(" ", "_")
        
#         with open(csv_path, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 if row["food"].lower().replace(" ", "_") == food_key:
#                     return True
        
#         return False
#     except:
#         return False


# def predict_food(image_path):
#     """
#     Advanced food prediction with:
#     1. Test-Time Augmentation (TTA)
#     2. Enhanced calibrated confidence
#     3. CSV availability boosting
#     4. Better uncertainty handling
#     5. FIXED: Proper bounds checking
#     """
#     if model is None:
#         return {
#             "food_name": "unknown",
#             "confidence": 0.0,
#             "is_confident": False,
#             "top_predictions": [],
#             "error": "Model not loaded"
#         }
    
#     try:
#         print(f"\n{'='*70}")
#         print(f"🔍 ADVANCED PREDICTION ANALYSIS")
#         print(f"{'='*70}")
        
#         # Step 1: Test-Time Augmentation
#         print(f"📸 Running predictions on multiple image variants...")
#         predictions = predict_with_tta(image_path, model)
        
#         if predictions is None:
#             return {
#                 "food_name": "unknown",
#                 "confidence": 0.0,
#                 "is_confident": False,
#                 "top_predictions": [],
#                 "error": "Image preprocessing failed"
#             }
        
#         # CRITICAL FIX: Verify predictions match labels length
#         num_classes = len(predictions)
#         num_labels = len(CLASS_LABELS)
        
#         print(f"🔍 Model outputs: {num_classes} classes")
#         print(f"🔍 Labels available: {num_labels} labels")
        
#         if num_classes != num_labels:
#             print(f"⚠️  WARNING: Mismatch between model ({num_classes}) and labels ({num_labels})")
#             # Truncate or pad predictions to match labels
#             if num_classes > num_labels:
#                 print(f"   Truncating predictions to {num_labels} classes")
#                 predictions = predictions[:num_labels]
#             else:
#                 print(f"   Padding predictions to {num_labels} classes")
#                 predictions = np.pad(predictions, (0, num_labels - num_classes), 'constant')
        
#         # Step 2: Get top predictions with BOUNDS CHECKING
#         top_k_indices = np.argsort(predictions)[-min(10, len(predictions)):][::-1]
        
#         # Step 3: Calculate confidence for each with SAFE INDEXING
#         predictions_with_confidence = []
#         for idx in top_k_indices:
#             # CRITICAL: Bounds check
#             if idx >= len(CLASS_LABELS):
#                 print(f"⚠️  Skipping index {idx} (out of bounds)")
#                 continue
            
#             food_name = CLASS_LABELS[idx].replace("_", " ")
#             raw_prob = float(predictions[idx])
            
#             # Calculate calibrated confidence
#             calibrated = get_calibrated_confidence(predictions)
            
#             # Check CSV availability
#             has_nutrition = check_csv_availability(food_name)
            
#             # Boost confidence if nutrition data available
#             final_confidence = calibrated
#             if has_nutrition:
#                 final_confidence = min(calibrated * 1.15, 1.0)
            
#             predictions_with_confidence.append({
#                 "food": food_name.title(),
#                 "raw_prob": raw_prob * 100,
#                 "calibrated": calibrated * 100,
#                 "final": final_confidence * 100,
#                 "has_nutrition": has_nutrition,
#                 "index": idx
#             })
        
#         # Check if we got any valid predictions
#         if not predictions_with_confidence:
#             print(f"❌ No valid predictions could be extracted")
#             return {
#                 "food_name": "unknown",
#                 "confidence": 0.0,
#                 "is_confident": False,
#                 "top_predictions": [],
#                 "error": "No valid predictions"
#             }
        
#         # Step 4: Select best prediction
#         best = predictions_with_confidence[0]
        
#         # Calculate entropy
#         entropy = calculate_entropy(predictions)
#         max_entropy = np.log(len(predictions))
#         normalized_entropy = entropy / max_entropy
        
#         # Debug output
#         print(f"\n📊 TTA Results:")
#         print(f"   Top prediction: {best['food']}")
#         print(f"   Raw probability: {best['raw_prob']:.2f}%")
#         print(f"   Calibrated confidence: {best['calibrated']:.2f}%")
#         print(f"   Final confidence: {best['final']:.2f}%")
#         print(f"   Has nutrition data: {'✅' if best['has_nutrition'] else '❌'}")
#         print(f"   Entropy: {entropy:.3f} (normalized: {normalized_entropy:.3f})")
        
#         print(f"\n🎯 Top 5 predictions:")
#         for i, pred in enumerate(predictions_with_confidence[:5], 1):
#             nutrition_marker = "✅" if pred['has_nutrition'] else "❌"
#             print(f"   {i}. {pred['food']}: {pred['final']:.2f}% {nutrition_marker}")
        
#         # Step 5: Determine if confident
#         is_confident = best['final'] >= (CONFIDENCE_THRESHOLD * 100)
        
#         print(f"\n{'='*70}")
#         print(f"✅ FINAL RESULT:")
#         print(f"   Food: {best['food']}")
#         print(f"   Confidence: {best['final']:.2f}%")
#         print(f"   Threshold: {CONFIDENCE_THRESHOLD * 100}%")
#         print(f"   Is confident: {is_confident}")
#         print(f"{'='*70}\n")
        
#         # Prepare top predictions for UI
#         top_predictions = [
#             {
#                 "food": p["food"],
#                 "confidence": round(p["raw_prob"], 2),
#                 "calibrated_confidence": round(p["final"], 2),
#                 "has_nutrition": p["has_nutrition"]
#             }
#             for p in predictions_with_confidence[:5]
#         ]
        
#         return {
#             "food_name": best['food'].lower().replace(" ", "_"),
#             "confidence": round(best['final'], 2),
#             "raw_confidence": round(best['raw_prob'], 2),
#             "is_confident": is_confident,
#             "top_predictions": top_predictions,
#             "entropy": round(entropy, 3),
#             "has_nutrition": best['has_nutrition']
#         }
        
#     except Exception as e:
#         print(f"❌ Prediction error: {e}")
#         import traceback
#         traceback.print_exc()
        
#         return {
#             "food_name": "unknown",
#             "confidence": 0.0,
#             "is_confident": False,
#             "top_predictions": [],
#             "error": str(e)
#         }


# # Legacy compatibility
# def softmax_with_temperature(logits, temperature=1.0):
#     """Temperature-scaled softmax"""
#     scaled_logits = logits / temperature
#     exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
#     return exp_logits / np.sum(exp_logits)



"""
CLIP-based Food Detection
Works for UNLIMITED foods - no training needed!
Just list the foods you want to detect
"""

import numpy as np
from PIL import Image
import os

# Try to import CLIP
try:
    import torch
    import clip
    CLIP_AVAILABLE = True
    
    # Load CLIP model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print(f"✅ CLIP model loaded on {device}")
    
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIDENCE_THRESHOLD = 0.30
# ==========================================

# ================= COMPREHENSIVE FOOD LIST =================
# Add ANY foods you want - CLIP can recognize them all!
FOOD_LIST = [
    # Indian Foods
    "samosa", "dosa", "idli", "vada", "biryani", "pakoda", "paneer tikka",
    "butter chicken", "palak paneer", "tandoori chicken", "naan bread",
    "roti", "paratha", "chapati", "puri", "bhaji", "pav bhaji",
    "chole", "dal", "rajma", "chicken curry", "fish curry", "korma",
    
    # Chinese Foods
    "dim sum", "baozi", "dumplings", "fried rice", "lo mein", "chow mein",
    "spring rolls", "egg rolls", "wonton soup", "hot and sour soup",
    "kung pao chicken", "sweet and sour pork", "peking duck", "congee",
    
    # Mexican Foods
    "tacos", "burritos", "enchiladas", "quesadilla", "nachos", "guacamole",
    "tamales", "fajitas", "tostadas", "chilaquiles", "mole", "pozole",
    
    # Italian Foods
    "pizza", "pasta", "lasagna", "risotto", "spaghetti", "ravioli",
    "gnocchi", "cannoli", "tiramisu", "panna cotta", "bruschetta",
    
    # American Foods
    "hamburger", "hot dog", "french fries", "fried chicken", "mac and cheese",
    "grilled cheese sandwich", "pancakes", "waffles", "donuts", "apple pie",
    
    # Japanese Foods
    "sushi", "sashimi", "ramen", "udon", "tempura", "teriyaki",
    "takoyaki", "okonomiyaki", "miso soup", "tonkatsu",
    
    # Middle Eastern Foods
    "falafel", "hummus", "shawarma", "kebab", "tabbouleh", "baba ganoush",
    
    # Thai Foods
    "pad thai", "green curry", "red curry", "tom yum soup", "papaya salad",
    
    # Breakfast Foods
    "eggs benedict", "omelette", "french toast", "breakfast burrito",
    "cereal", "yogurt", "smoothie bowl",
    
    # Desserts
    "ice cream", "cake", "cookies", "brownies", "cupcakes", "cheesecake",
    "chocolate mousse", "pudding", "custard", "macarons",
    
    # Beverages/Soups
    "soup", "salad", "sandwich", "wrap", "bowl",
    
    # Add more as needed...
]
# ==========================================================


def check_csv_availability(food_name):
    """Check if food exists in CSV database"""
    try:
        import csv
        from django.conf import settings
        
        csv_path = os.path.join(settings.BASE_DIR, "data", "calories.csv")
        food_key = food_name.lower().replace(" ", "_")
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["food"].lower().replace(" ", "_") == food_key:
                    return True
        return False
    except:
        return False


def predict_food(image_path):
    """
    Predict food using CLIP - works for ANY food in FOOD_LIST
    No training needed!
    """
    if not CLIP_AVAILABLE:
        return {
            "food_name": "unknown",
            "confidence": 0.0,
            "is_confident": False,
            "top_predictions": [],
            "error": "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
        }
    
    try:
        print(f"\n{'='*70}")
        print(f"🔍 CLIP FOOD DETECTION (Unlimited Foods)")
        print(f"{'='*70}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        # Create text prompts
        text_prompts = [f"a photo of {food}" for food in FOOD_LIST]
        text_inputs = clip.tokenize(text_prompts).to(device)
        
        # Get predictions
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_inputs)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(10)
        
        # Extract top predictions
        results = []
        for i, (value, idx) in enumerate(zip(values, indices)):
            food = FOOD_LIST[idx]
            confidence = float(value) * 100
            has_nutrition = check_csv_availability(food)
            
            results.append({
                "food": food.title(),
                "confidence": confidence,
                "has_nutrition": has_nutrition,
                "rank": i + 1
            })
        
        # Best prediction
        best = results[0]
        
        print(f"\n📊 CLIP Results:")
        print(f"   Top prediction: {best['food']}")
        print(f"   Confidence: {best['confidence']:.2f}%")
        print(f"   Has nutrition: {'✅' if best['has_nutrition'] else '❌'}")
        
        print(f"\n🎯 Top 5 predictions:")
        for pred in results[:5]:
            nutrition = "✅" if pred['has_nutrition'] else "❌"
            print(f"   {pred['rank']}. {pred['food']}: {pred['confidence']:.2f}% {nutrition}")
        
        is_confident = best['confidence'] >= CONFIDENCE_THRESHOLD * 100
        
        print(f"\n{'='*70}")
        print(f"✅ FINAL RESULT:")
        print(f"   Food: {best['food']}")
        print(f"   Confidence: {best['confidence']:.2f}%")
        print(f"   Is confident: {is_confident}")
        print(f"{'='*70}\n")
        
        # Format for UI
        top_predictions = [
            {
                "food": p["food"],
                "confidence": round(p["confidence"], 2),
                "calibrated_confidence": round(p["confidence"], 2),
                "has_nutrition": p["has_nutrition"]
            }
            for p in results[:5]
        ]
        
        return {
            "food_name": best['food'].lower().replace(" ", "_"),
            "confidence": round(best['confidence'], 2),
            "raw_confidence": round(best['confidence'], 2),
            "is_confident": is_confident,
            "top_predictions": top_predictions,
            "has_nutrition": best['has_nutrition'],
            "method": "CLIP"
        }
        
    except Exception as e:
        print(f"❌ CLIP prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "food_name": "unknown",
            "confidence": 0.0,
            "is_confident": False,
            "top_predictions": [],
            "error": str(e)
        }


# Legacy compatibility
def softmax_with_temperature(logits, temperature=1.0):
    """Not used in CLIP version"""
    pass
