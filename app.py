from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load models and data at startup
MODEL_DIR = "models"
try:
    knn_model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    hotel_catalog = pd.read_pickle(os.path.join(MODEL_DIR, "hotel_catalog.pkl"))
    
    # Extract unique values for the frontend dropdowns
    unique_countries = sorted(hotel_catalog['country'].unique().tolist())
    unique_hotels = sorted(hotel_catalog['hotel'].unique().tolist())
except Exception as e:
    print(f"Error loading models. Please ensure train.py was run successfully. Error: {e}")
    knn_model, preprocessor, hotel_catalog = None, None, None
    unique_countries, unique_hotels = [], []

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", 
                           countries=unique_countries, 
                           hotels=unique_hotels, 
                           recommendations=None)

@app.route("/recommend", methods=["POST"])
def recommend():
    if not knn_model or not preprocessor or hotel_catalog is None:
        return render_template("index.html", error="Model not loaded. Please train the model first.", countries=unique_countries, hotels=unique_hotels)

    try:
        # Extract user preferences
        selected_country = request.form.get("country")
        selected_hotel = request.form.get("hotel_type")
        target_adr = float(request.form.get("price", 100))
        target_rating = float(request.form.get("rating", 4.0))

        # Create a dataframe for the user's input
        input_data = pd.DataFrame([{
            'hotel': selected_hotel,
            'country': selected_country,
            'adr': target_adr,
            'rating': target_rating
        }])

        # Preprocess input data
        processed_input = preprocessor.transform(input_data)

        # Get top 5 recommendations
        distances, indices = knn_model.kneighbors(processed_input, n_neighbors=5)
        
        # Extract the recommended hotels from the catalog
        recommended_indices = indices[0]
        recommendations_df = hotel_catalog.iloc[recommended_indices]
        
        # Add distance as a score (optional, lower is better)
        recommendations_df = recommendations_df.copy()
        recommendations_df['match_score'] = distances[0]
        
        # Convert to a list of dicts for rendering
        recs = recommendations_df.to_dict(orient='records')
        
        return render_template("index.html", 
                               countries=unique_countries, 
                               hotels=unique_hotels, 
                               recommendations=recs,
                               selected_country=selected_country,
                               selected_hotel=selected_hotel,
                               target_adr=target_adr,
                               target_rating=target_rating)
                               
    except Exception as e:
        return render_template("index.html", error=str(e), countries=unique_countries, hotels=unique_hotels)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
