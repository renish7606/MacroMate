"""
Model Diagnostic Script for Food Detection
Run this to test your model and identify issues
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Add project to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("="*70)
print("🔍 FOOD DETECTION MODEL DIAGNOSTICS")
print("="*70)

# ============================================
# 1. CHECK MODEL FILE
# ============================================
print("\n1️⃣ Checking model file...")
MODEL_PATH = os.path.join(BASE_DIR, "models", "food101_efficientnet.keras")

if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
    print(f"✅ Model file found: {MODEL_PATH}")
    print(f"   File size: {file_size:.2f} MB")
else:
    print(f"❌ Model file NOT found at: {MODEL_PATH}")
    print("   Please ensure your model is in the correct location")
    sys.exit(1)

# ============================================
# 2. LOAD MODEL
# ============================================
print("\n2️⃣ Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
    
    # Print model architecture
    print(f"\n📊 Model Architecture:")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total params: {model.count_params():,}")
    
    # Check if output is correct size
    expected_classes = 101  # Food-101 dataset
    actual_classes = model.output_shape[-1]
    if actual_classes == expected_classes:
        print(f"✅ Output classes: {actual_classes} (correct for Food-101)")
    else:
        print(f"⚠️  Output classes: {actual_classes} (expected {expected_classes})")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# 3. CHECK LABELS FILE
# ============================================
print("\n3️⃣ Checking labels file...")
LABELS_PATH = os.path.join(BASE_DIR, "data", "food101_labels.txt")

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✅ Labels file found: {LABELS_PATH}")
    print(f"   Number of labels: {len(labels)}")
    print(f"   First 5 labels: {labels[:5]}")
    print(f"   Last 5 labels: {labels[-5:]}")
    
    if len(labels) != model.output_shape[-1]:
        print(f"⚠️  WARNING: Number of labels ({len(labels)}) doesn't match model output ({model.output_shape[-1]})")
else:
    print(f"❌ Labels file NOT found at: {LABELS_PATH}")
    sys.exit(1)

# ============================================
# 4. TEST WITH RANDOM IMAGE
# ============================================
print("\n4️⃣ Testing prediction with random input...")

# Create a random test image
test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
test_image_pil = Image.fromarray(test_image)

# Preprocess
test_input = np.array(test_image_pil, dtype=np.float32)
test_input = tf.keras.applications.efficientnet.preprocess_input(test_input)
test_input = np.expand_dims(test_input, axis=0)

# Predict
predictions = model.predict(test_input, verbose=0)[0]

print(f"\n📊 Prediction Statistics:")
print(f"   Output shape: {predictions.shape}")
print(f"   Min value: {predictions.min():.6f}")
print(f"   Max value: {predictions.max():.6f}")
print(f"   Mean value: {predictions.mean():.6f}")
print(f"   Sum: {predictions.sum():.6f}")

# Check if output is logits or probabilities
if 0.99 < predictions.sum() < 1.01:
    print(f"✅ Output appears to be probabilities (sum ≈ 1)")
    probs = predictions
else:
    print(f"⚠️  Output appears to be logits (sum = {predictions.sum():.2f})")
    print(f"   Applying softmax...")
    probs = tf.nn.softmax(predictions).numpy()

# Show top predictions
top_indices = np.argsort(probs)[-5:][::-1]
print(f"\n🎯 Top 5 predictions for random image:")
for i, idx in enumerate(top_indices, 1):
    print(f"   {i}. {labels[idx]}: {probs[idx]*100:.2f}%")

# Calculate entropy
def calculate_entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))

entropy = calculate_entropy(probs)
max_entropy = np.log(len(probs))
normalized_entropy = entropy / max_entropy

print(f"\n📈 Uncertainty metrics:")
print(f"   Entropy: {entropy:.3f}")
print(f"   Max possible entropy: {max_entropy:.3f}")
print(f"   Normalized entropy: {normalized_entropy:.3f} (0=certain, 1=completely uncertain)")
print(f"   Top probability: {probs.max()*100:.2f}%")
print(f"   Gap to 2nd: {(probs[top_indices[0]] - probs[top_indices[1]])*100:.2f}%")

# ============================================
# 5. DIAGNOSIS
# ============================================
print("\n" + "="*70)
print("🩺 DIAGNOSIS")
print("="*70)

issues = []
warnings = []

# Check 1: Model outputs
if predictions.sum() > 10:
    issues.append("Model outputs appear to be unnormalized logits")
    print("❌ Model is outputting logits (raw scores) instead of probabilities")
    print("   → Your code should apply softmax: tf.nn.softmax(predictions)")

# Check 2: Always high confidence
if probs.max() > 0.95:
    warnings.append("Very high confidence on random image")
    print("⚠️  Model shows very high confidence (>95%) even on random noise")
    print("   This suggests:")
    print("   • Model might be overfitted")
    print("   • Training data might have been too similar")
    print("   • Model architecture might not match saved weights")

# Check 3: Distribution
if normalized_entropy < 0.1:
    warnings.append("Very low entropy (over-confident)")
    print("⚠️  Very low entropy indicates over-confidence")
    print("   • Model is too certain about its predictions")
    print("   • This happens with overfitted or mismatched models")

# ============================================
# 6. RECOMMENDATIONS
# ============================================
print("\n" + "="*70)
print("💡 RECOMMENDATIONS")
print("="*70)

print("\n1. Confidence Calibration:")
print("   ✓ Use the calibrated confidence calculation in the fixed ml_food_predictor.py")
print("   ✓ Don't rely on raw softmax probabilities alone")
print("   ✓ Consider entropy and gap between top predictions")

print("\n2. Threshold Adjustment:")
print("   ✓ Lower your confidence threshold from 40% to 20-30%")
print("   ✓ More predictions will require manual confirmation, but that's OK")

print("\n3. Model Retraining (if possible):")
print("   ✓ Add data augmentation during training")
print("   ✓ Use label smoothing")
print("   ✓ Add dropout layers")
print("   ✓ Validate on truly unseen data")

print("\n4. Alternative Approach:")
print("   ✓ Use an ensemble of models")
print("   ✓ Combine ML prediction with keyword matching from image filename")
print("   ✓ Always show top 3 predictions for user to choose")

# ============================================
# 7. TEST WITH ACTUAL IMAGE (if available)
# ============================================
print("\n" + "="*70)
print("📸 TESTING WITH REAL IMAGE")
print("="*70)

# Check for test images
test_dirs = ["uploads", "static/uploads", "media/uploads"]
test_image_path = None

for test_dir in test_dirs:
    full_path = os.path.join(BASE_DIR, test_dir)
    if os.path.exists(full_path):
        images = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            test_image_path = os.path.join(full_path, images[0])
            break

if test_image_path and os.path.exists(test_image_path):
    print(f"\n✅ Found test image: {test_image_path}")
    
    # Load and preprocess
    img = Image.open(test_image_path).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    real_predictions = model.predict(img_array, verbose=0)[0]
    
    if not (0.99 < real_predictions.sum() < 1.01):
        real_predictions = tf.nn.softmax(real_predictions).numpy()
    
    # Show results
    top_indices = np.argsort(real_predictions)[-5:][::-1]
    print(f"\n🎯 Top 5 predictions for real image:")
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {labels[idx]}: {real_predictions[idx]*100:.2f}%")
    
    # Calculate metrics
    entropy = calculate_entropy(real_predictions)
    normalized_entropy = entropy / max_entropy
    
    print(f"\n📊 Confidence metrics:")
    print(f"   Top probability: {real_predictions.max()*100:.2f}%")
    print(f"   Gap to 2nd: {(real_predictions[top_indices[0]] - real_predictions[top_indices[1]])*100:.2f}%")
    print(f"   Normalized entropy: {normalized_entropy:.3f}")
    
    # Calibrated confidence
    top_prob = real_predictions[top_indices[0]]
    gap = real_predictions[top_indices[0]] - real_predictions[top_indices[1]]
    calibrated = top_prob * (1 + gap) * (1 - normalized_entropy * 0.5)
    
    print(f"\n✨ Calibrated confidence: {calibrated*100:.2f}%")
    print(f"   (This is what user should see)")
    
else:
    print("\n⚠️  No test images found in uploads directory")
    print("   Upload an image through the web interface to test with real data")

print("\n" + "="*70)
print("✅ Diagnostics complete!")
print("="*70)
print("\nNext steps:")
print("1. Replace detector/ml_food_predictor.py with the fixed version")
print("2. Replace detector/views.py with the fixed version")
print("3. Test with several different food images")
print("4. Adjust CONFIDENCE_THRESHOLD if needed (start with 0.25)")
print("="*70)
