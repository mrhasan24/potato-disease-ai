# PotatoGuard AI — Potato Disease Detection System

A simple, professional Flask web application that uses a trained TensorFlow
(Keras) MobileNet model to classify potato leaf diseases from an uploaded
image.

BSc Thesis Demonstration Project.

---

## Features

- Upload a potato leaf image (PNG/JPG) via click or drag-and-drop
- Live image preview before prediction
- AI prediction using your trained `best_MobileNet.keras` model
- Confidence score for the top prediction
- Top-3 predicted classes with confidence bars
- Clean, responsive, green agriculture-themed Bootstrap 5 UI
- Basic error handling and logging

## Detected Classes

| # | Class | Meaning |
|---|-------|---------|
| 0 | Potato___bacterial_wilt | Bacterial Wilt |
| 1 | Potato___early_blight | Early Blight |
| 2 | Potato___healthy | Healthy |
| 3 | Potato___late_blight | Late Blight |
| 4 | Potato___leafroll_virus | Leafroll Virus |
| 5 | Potato___mosaic_virus | Mosaic Virus |
| 6 | Potato___pests | Pest Damage |
| 7 | Potato___phytophthora | Phytophthora |

---

## Project Structure

```
potato_disease_app/
├── app.py                  # Flask application (routes, model loading, prediction logic)
├── requirements.txt        # Python dependencies
├── model/
│   └── best_MobileNet.keras   # Your trained model (already included)
├── templates/
│   └── index.html          # Main UI page
├── static/
│   ├── css/
│   │   └── style.css       # Custom green theme styling
│   └── uploads/            # Uploaded images are saved here at runtime
└── README.md
```

---

## Setup Instructions (Windows)

### 1. Install Python

Make sure you have **Python 3.10 or 3.11** installed (TensorFlow does not yet
support the very latest Python versions on Windows). Check with:

```
python --version
```

### 2. Create a virtual environment (recommended)

Open **Command Prompt** or **PowerShell** in the project folder:

```
cd path\to\potato_disease_app
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

This installs Flask, TensorFlow, OpenCV, Pillow, NumPy, and Werkzeug.

> Installing TensorFlow can take a few minutes the first time.

### 4. Run the application

```
python app.py
```

You should see output similar to:

```
Loading model from: ...\model\best_MobileNet.keras
Model loaded successfully.
 * Running on http://127.0.0.1:5000
```

### 5. Open in your browser

Go to: **http://127.0.0.1:5000**

Upload a potato leaf image, click **Predict Disease**, and view the result.

---

## Important Note on Image Preprocessing

The app preprocesses every uploaded image exactly like this before feeding it
to the model:

1. Read image with OpenCV
2. Convert BGR → RGB
3. Resize to **224×224**
4. Scale pixel values to the range **[0, 1]** (divide by 255)
5. Add a batch dimension

If your model was trained with a **different** preprocessing scheme (for
example `tf.keras.applications.mobilenet.preprocess_input`, which scales to
**[-1, 1]** instead of [0, 1]), predictions may be inaccurate. In that case,
open `app.py` and update the `preprocess_image()` function to match exactly
how your training pipeline preprocessed images.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install -r requirements.txt` inside the activated virtual environment. |
| Model fails to load on startup | Confirm `model/best_MobileNet.keras` exists and was not corrupted during transfer. |
| Predictions look wrong / low confidence everywhere | Check the preprocessing note above — it must match your training pipeline. |
| Port 5000 already in use | Change the port in the last line of `app.py`, e.g. `app.run(debug=True, port=5001)`. |
| Uploaded image doesn't show a preview | Make sure it's a valid PNG or JPG under 8 MB. |

---

## Tech Stack

- **Backend:** Python, Flask
- **AI/ML:** TensorFlow (Keras), MobileNet architecture
- **Image Processing:** OpenCV, Pillow
- **Frontend:** HTML5, Bootstrap 5, vanilla JavaScript, CSS3
- **No database required** — this is a stateless, single-prediction demo app.

---

© 2026 PotatoGuard AI — BSc Thesis Demonstration Project.
