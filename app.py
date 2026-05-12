import subprocess
import sys
import os

# ── Step 1: Install all dependencies ──────────────────────────────────────────
print("[STARTUP] Installing dependencies...")
_base = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r",
     os.path.join(_base, "requirements.txt"), "--quiet",
     "--break-system-packages"]
)
print("[STARTUP] Dependencies installed.")

# ── Step 2: Train models fresh (avoids pickle version mismatch on Render) ─────
print("[STARTUP] Training model from CSV data...")
subprocess.check_call([sys.executable, os.path.join(_base, "train.py")], cwd=_base)
print("[STARTUP] Training complete.")

# ── Step 3: Load trained models ───────────────────────────────────────────────
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
MODEL_DIR = os.path.join(_base, "models")

try:
    knn_model     = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
    preprocessor  = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    hotel_catalog = joblib.load(os.path.join(MODEL_DIR, "hotel_catalog.pkl"))

    unique_countries = sorted(hotel_catalog['country'].unique().tolist())
    unique_hotels    = sorted(hotel_catalog['hotel'].unique().tolist())
    print(f"[STARTUP] Models loaded. {len(unique_countries)} countries, {len(unique_hotels)} hotel types.")
except Exception as e:
    print(f"[STARTUP ERROR] Failed to load models: {e}")
    knn_model = preprocessor = hotel_catalog = None
    unique_countries = unique_hotels = []

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           countries=unique_countries,
                           hotels=unique_hotels,
                           recommendations=None)

@app.route("/recommend", methods=["POST"])
def recommend():
    if knn_model is None or preprocessor is None or hotel_catalog is None:
        return render_template("index.html",
                               error="Models not loaded. Please check server logs.",
                               countries=unique_countries,
                               hotels=unique_hotels)
    try:
        selected_country = request.form.get("country")
        selected_hotel   = request.form.get("hotel_type")
        target_adr       = float(request.form.get("price", 12500))
        target_rating    = float(request.form.get("rating", 4.0))
        booking_date     = request.form.get("booking_date", "")
        num_persons      = request.form.get("num_persons", "2")

        input_data = pd.DataFrame([{
            'hotel':   selected_hotel,
            'country': selected_country,
            'adr':     target_adr,
            'rating':  target_rating
        }])

        processed_input    = preprocessor.transform(input_data)
        distances, indices = knn_model.kneighbors(processed_input, n_neighbors=5)

        recommendations_df = hotel_catalog.iloc[indices[0]].copy()
        recommendations_df['match_score'] = distances[0]
        recs = recommendations_df.to_dict(orient='records')

        return render_template("index.html",
                               countries=unique_countries,
                               hotels=unique_hotels,
                               recommendations=recs,
                               selected_country=selected_country,
                               selected_hotel=selected_hotel,
                               target_adr=target_adr,
                               target_rating=target_rating,
                               booking_date=booking_date,
                               num_persons=num_persons)
    except Exception as e:
        return render_template("index.html",
                               error=str(e),
                               countries=unique_countries,
                               hotels=unique_hotels)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
