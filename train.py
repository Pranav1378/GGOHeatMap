#!/usr/bin/env python3
"""
train.py - Simplified training script for Great Gray Owl habitat modeling
Creates synthetic but realistic training data and saves model artifacts.
"""

import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

def main():
    print("🦉 Training Great Gray Owl Habitat Model...")
    
    # 1. Create synthetic owl observations (fallback if API fails)
    print("▸ Creating training data...")
    
    # Realistic owl locations in Yosemite (higher elevations, forest areas)
    owl_locations = [
        (37.8742, -119.3514),  # Tuolumne Meadows
        (37.9469, -119.7911),  # Hetch Hetchy
        (37.8500, -119.4000),  # High country
        (37.7800, -119.7000),  # Wawona area
        (37.8200, -119.5500),  # Mid-elevation forest
    ]
    
    # Add some noise to create more observations
    np.random.seed(42)
    all_obs = []
    for lat, lon in owl_locations:
        for i in range(4):  # 4 observations per location
            new_lat = lat + np.random.normal(0, 0.01)
            new_lon = lon + np.random.normal(0, 0.01)
            all_obs.append((new_lat, new_lon))
    
    # 2. Feature extraction function
    def extract_features(lat, lon):
        valley_lat, valley_lon = 37.7459, -119.5936
        dist_to_valley = np.sqrt((lat - valley_lat)**2 + (lon - valley_lon)**2)
        
        elevation = 1200 + dist_to_valley * 1500 + np.random.normal(0, 100)
        slope = min(45, dist_to_valley * 20) + np.random.normal(0, 3)
        d_road = abs(lat - 37.75) * 5000 + abs(lon + 119.6) * 3000 + np.random.normal(0, 500)
        d_water = abs(lat - 37.76) * 2000 + abs(lon + 119.58) * 2500 + np.random.normal(0, 300)
        
        return [max(1000, elevation), max(0, slope), max(0, d_road), max(0, d_water)]
    
    # 3. Create training dataset
    print("▸ Building training set...")
    
    # Presence points
    presence_data = []
    for lat, lon in all_obs:
        features = extract_features(lat, lon)
        presence_data.append(features + [1])
    
    # Background points
    background_data = []
    for _ in range(200):
        lat = np.random.uniform(37.5, 38.2)
        lon = np.random.uniform(-120, -119)
        features = extract_features(lat, lon)
        background_data.append(features + [0])
    
    # Combine
    all_data = presence_data + background_data
    df = pd.DataFrame(all_data, columns=['elevation', 'slope', 'd_road', 'd_water', 'label'])
    
    # 4. Train model
    print("▸ Training Random Forest...")
    X = df[['elevation', 'slope', 'd_road', 'd_water']].values
    y = df['label'].values
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    print(f"  Model accuracy: {model.score(X, y):.3f}")
    
    # 5. Create prediction grid
    print("▸ Creating prediction grid...")
    grid_points = []
    
    for lat in np.linspace(37.5, 38.2, 40):
        for lon in np.linspace(-120, -119, 40):
            features = extract_features(lat, lon)
            prob = model.predict_proba([features])[0, 1]
            
            grid_points.append({
                'lat': lat, 'lon': lon,
                'elev': features[0], 'slope': features[1],
                'd_road': features[2], 'd_water': features[3],
                'prob': prob
            })
    
    grid_df = pd.DataFrame(grid_points)
    obs_df = pd.DataFrame(all_obs, columns=['lat', 'lon'])
    
    # 6. Save everything
    print("▸ Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    grid_df.to_csv('grid.csv', index=False)
    obs_df.to_csv('obs.csv', index=False)
    
    print("✅ Training complete!")
    print(f"   - {len(obs_df)} owl observations")
    print(f"   - {len(grid_df)} prediction grid points")
    print(f"   - Model saved to model.pkl")

if __name__ == "__main__":
    main()
