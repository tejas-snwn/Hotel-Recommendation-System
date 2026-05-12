import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pycountry

def get_country_name(code):
    try:
        country = pycountry.countries.get(alpha_3=code)
        if country:
            return country.name
        country = pycountry.countries.get(alpha_2=code)
        if country:
            return country.name
    except Exception:
        pass
    return code

def generate_hotel_dataset(input_csv):
    print("Loading dataset...")
    df = pd.read_csv(input_csv)

    df = df.dropna(subset=['country'])
    df = df[df['adr'] > 0]

    print("Converting country codes to full names and ADR to INR...")
    df['country'] = df['country'].apply(get_country_name)
    df['adr'] = df['adr'] * 83.0  # Convert USD to INR

    print("Aggregating into unique hotels...")
    grouped = df.groupby(['hotel', 'country']).agg(
        adr=('adr', 'mean'),
        total_bookings=('adr', 'count')
    ).reset_index()

    grouped = grouped[grouped['total_bookings'] > 5].copy()

    np.random.seed(42)
    grouped['rating'] = np.round(np.random.uniform(3.0, 5.0, size=len(grouped)), 1)
    grouped['hotel_name'] = grouped['hotel'] + " in " + grouped['country']

    print(f"Generated {len(grouped)} unique hotels.")
    return grouped

def train_model(hotel_df, output_dir):
    print("Preprocessing data and encoding features...")

    categorical_features = ['hotel', 'country']
    numerical_features   = ['adr', 'rating']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(),              numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    X = preprocessor.fit_transform(hotel_df)

    print("Training KNN model...")
    knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    knn.fit(X)

    print("Saving model artifacts...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(knn,          os.path.join(output_dir, 'knn_model.pkl'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    joblib.dump(hotel_df,     os.path.join(output_dir, 'hotel_catalog.pkl'))  # joblib, NOT pd.to_pickle

    print("Model training pipeline completed successfully!")

if __name__ == "__main__":
    csv_path = "hotel_bookings.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
    else:
        hotel_data = generate_hotel_dataset(csv_path)
        train_model(hotel_data, output_dir="models")
