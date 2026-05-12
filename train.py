import pandas as pd
import numpy as np
import joblib
import os
import random
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

    print("Generating random hotel names and expanding dataset...")
    prefixes = ["The Grand", "Royal", "Sunset", "Ocean", "Majestic", "Crystal",
                "Emerald", "Golden", "Luxury", "Elite", "Classic", "Urban",
                "Imperial", "Sapphire", "Pearl", "Tranquil", "Harmony", "Serene", "Opal"]
    suffixes = ["Palace", "Resort & Spa", "Suites", "Inn", "Boutique Hotel",
                "Retreat", "Oasis", "Plaza", "Residency", "Grand", "Towers",
                "View", "Heights", "Manor", "Villa", "Lodge", "Residences"]

    expanded_rows = []
    random.seed(42)
    np.random.seed(42)

    for _, row in grouped.iterrows():
        for _ in range(5):
            new_row = row.copy()
            new_row['hotel_name'] = f"{random.choice(prefixes)} {random.choice(suffixes)}"
            noise_factor = random.uniform(0.8, 1.2)
            new_row['adr'] = round(row['adr'] * noise_factor, 2)
            new_row['rating'] = round(random.uniform(3.0, 5.0), 1)
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    print(f"Generated {len(expanded_df)} unique hotels.")
    return expanded_df

def train_model(hotel_df, output_dir):
    print("Preprocessing data and encoding features...")

    categorical_features = ['hotel', 'country']
    numerical_features   = ['adr', 'rating']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(),                       numerical_features),
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
    joblib.dump(hotel_df,     os.path.join(output_dir, 'hotel_catalog.pkl'))

    print("Model training pipeline completed successfully!")

if __name__ == "__main__":
    csv_path = "hotel_bookings.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
    else:
        hotel_data = generate_hotel_dataset(csv_path)
        train_model(hotel_data, output_dir="models")
