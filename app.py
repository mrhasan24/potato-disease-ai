"""
Potato Disease Detection System
--------------------------------
A Flask web application that uses a trained TensorFlow (Keras) MobileNet
model to classify potato leaf diseases from an uploaded image.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import io
import time
import logging
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_MobileNet.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)          # Must match the model's expected input size
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload size

# Class names in the EXACT order the model was trained on
# (index position = model output index)
CLASS_NAMES = [
    "Potato___bacterial_wilt",
    "Potato___early_blight",
    "Potato___healthy",
    "Potato___late_blight",
    "Potato___leafroll_virus",
    "Potato___mosaic_virus",
    "Potato___pests",
    "Potato___phytophthora",
]

# Human-friendly display names + short descriptions for the UI
CLASS_INFO = {
    "Potato___bacterial_wilt": {
        "display_name": "Bacterial Wilt",
        "status": "disease",
        "description": "A bacterial infection that causes wilting, yellowing, "
                        "and eventual collapse of the plant due to vascular blockage.",
    },
    "Potato___early_blight": {
        "display_name": "Early Blight",
        "status": "disease",
        "description": "A fungal disease causing dark concentric-ring spots on "
                        "older leaves, often leading to premature leaf drop.",
    },
    "Potato___healthy": {
        "display_name": "Healthy",
        "status": "healthy",
        "description": "No signs of disease detected. The leaf appears healthy.",
    },
    "Potato___late_blight": {
        "display_name": "Late Blight",
        "status": "disease",
        "description": "A fast-spreading fungal-like disease causing dark, "
                        "water-soaked lesions; historically responsible for the "
                        "Irish potato famine.",
    },
    "Potato___leafroll_virus": {
        "display_name": "Leafroll Virus",
        "status": "disease",
        "description": "A viral infection causing upward rolling and stiffening "
                        "of leaves, often paired with stunted growth.",
    },
    "Potato___mosaic_virus": {
        "display_name": "Mosaic Virus",
        "status": "disease",
        "description": "A viral infection producing a mottled light/dark green "
                        "mosaic pattern on leaves and reduced yield.",
    },
    "Potato___pests": {
        "display_name": "Pest Damage",
        "status": "disease",
        "description": "Visible leaf damage caused by insect pests rather than "
                        "a pathogen (e.g. chewing or sucking insects).",
    },
    "Potato___phytophthora": {
        "display_name": "Phytophthora",
        "status": "disease",
        "description": "An oomycete (water mold) infection causing irregular "
                        "brown lesions and rapid tissue decay in humid conditions.",
    },
}

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("potato_disease_app")

# --------------------------------------------------------------------------
# Flask app setup
# --------------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------------------------------
# Load the trained model once at startup
# --------------------------------------------------------------------------
model = None
try:
    logger.info("Loading model from: %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as exc:  # noqa: BLE001
    logger.error("Failed to load model: %s", exc)
    model = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk with OpenCV and preprocess it exactly the way
    the model expects:
      1. Read as BGR (OpenCV default)
      2. Convert to RGB
      3. Resize to the model's input size (224x224)
      4. Scale pixel values to [0, 1]
      5. Add a batch dimension
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the uploaded image. The file may be corrupted.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_disease(image_path: str):
    """Run the model on a preprocessed image and return structured results."""
    if model is None:
        raise RuntimeError("Model is not loaded. Check server logs for details.")

    processed = preprocess_image(image_path)
    predictions = model.predict(processed, verbose=0)[0]  # shape: (num_classes,)

    # Sort class indices by confidence, descending
    ranked_indices = np.argsort(predictions)[::-1]

    top3 = []
    for idx in ranked_indices[:3]:
        class_key = CLASS_NAMES[idx]
        info = CLASS_INFO[class_key]
        top3.append({
            "class_name": class_key,
            "display_name": info["display_name"],
            "confidence": round(float(predictions[idx]) * 100, 2),
        })

    best_idx = int(ranked_indices[0])
    best_key = CLASS_NAMES[best_idx]
    best_info = CLASS_INFO[best_key]

    return {
        "predicted_class": best_key,
        "display_name": best_info["display_name"],
        "status": best_info["status"],
        "description": best_info["description"],
        "confidence": round(float(predictions[best_idx]) * 100, 2),
        "top3": top3,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the main upload/landing page."""
    return render_template("index.html", model_loaded=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction results as JSON."""
    if model is None:
        return jsonify({"success": False, "error": "Model is not loaded on the server."}), 500

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload a PNG or JPG image.",
        }), 400

    try:
        # Build a unique, safe filename so uploads never collide
        original_name = secure_filename(file.filename)
        unique_name = f"{int(time.time() * 1000)}_{original_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        logger.info("Saved upload to %s", save_path)

        result = predict_disease(save_path)
        result["success"] = True
        result["image_url"] = url_for("static", filename=f"uploads/{unique_name}")

        logger.info(
            "Prediction complete: %s (%.2f%%)",
            result["display_name"], result["confidence"]
        )

        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Invalid image uploaded: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "File is too large. Maximum size is 8 MB."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
