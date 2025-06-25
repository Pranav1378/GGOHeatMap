"""
Great Gray Owl Habitat Suitability - Streamlit App
--------------------------------------------------
Interactive web app showing habitat predictions for Great Gray Owls in Yosemite NP.
Loads pre-trained model and prediction grid from CSV files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
from pathlib import Path
import requests
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(layout="wide", page_title="🦉 Great Gray Owl Habitat")
st.title("🦉 Great Gray Owl – Habitat Suitability in Yosemite NP")

@st.cache_data
def load_data():
    """Load model artifacts (will be created by train.py during build)"""
    try:
        # Try to load pre-trained artifacts
        model = joblib.load("model.pkl")
        grid = pd.read_csv("grid.csv")
        obs = pd.read_csv("obs.csv")
        return model, grid, obs
    except FileNotFoundError:
        st.error("Model files not found. Please run train.py first.")
        st.stop()

# Load model and data
model, grid_data, obs_data = load_data()

# Simple feature extraction function (matches training)
def extract_features(lat, lon):
    """Extract features for a given location (simplified version)"""
    # Use distance from Yosemite Valley center as proxy for elevation
    valley_lat, valley_lon = 37.7459, -119.5936
    dist_to_valley = np.sqrt((lat - valley_lat)**2 + (lon - valley_lon)**2)
    
    # Simple feature approximations based on location
    elev = 1200 + dist_to_valley * 1500  # Elevation increases with distance from valley
    slope = min(45, dist_to_valley * 20)  # Slope increases with distance
    d_road = abs(lat - 37.75) * 5000 + abs(lon + 119.6) * 3000  # Distance to main roads
    d_water = abs(lat - 37.76) * 2000 + abs(lon + 119.58) * 2500  # Distance to rivers
    
    return np.array([[elev, slope, d_road, d_water]])

# UI Layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("🎯 Try Your Location")
    
    # Input controls
    lat = st.number_input(
        "Latitude", 
        min_value=37.5, 
        max_value=38.2, 
        value=37.85, 
        format="%.6f",
        help="Enter latitude between 37.5 and 38.2 (Yosemite area)"
    )
    
    lon = st.number_input(
        "Longitude", 
        min_value=-120.0, 
        max_value=-119.0, 
        value=-119.55, 
        format="%.6f",
        help="Enter longitude between -120.0 and -119.0 (Yosemite area)"
    )
    
    if st.button("🔮 Predict Habitat Suitability", type="primary"):
        # Extract features and make prediction
        features = extract_features(lat, lon)
        prob = model.predict_proba(features)[0, 1]
        
        # Display results
        st.success(f"**Habitat Suitability: {prob:.1%}**")
        
        # Feature breakdown
        elev, slope, d_road, d_water = features[0]
        st.info(f"""
        **Location Analysis:**
        - 🏔️ Elevation: {elev:.0f} m
        - 📐 Slope: {slope:.1f}°
        - 🛣️ Distance to roads: {d_road/1000:.2f} km
        - 💧 Distance to water: {d_water/1000:.2f} km
        """)
        
        # Interpretation
        if prob > 0.7:
            st.success("🟢 **Excellent habitat** - High owl presence probability!")
        elif prob > 0.4:
            st.warning("🟡 **Moderate habitat** - Some potential for owl presence")
        else:
            st.error("🔴 **Poor habitat** - Low owl presence probability")
    
    # Add some example locations
    st.subheader("📍 Try These Locations")
    example_locations = {
        "Yosemite Valley": (37.7459, -119.5936),
        "Half Dome Area": (37.7459, -119.5333),
        "Tuolumne Meadows": (37.8742, -119.3514),
        "Hetch Hetchy": (37.9469, -119.7911)
    }
    
    for name, (example_lat, example_lon) in example_locations.items():
        if st.button(f"📌 {name}", key=name):
            st.rerun()

with col2:
    st.subheader("🗺️ Interactive Habitat Map")
    
    # Prepare grid data for visualization
    if not grid_data.empty:
        # Color mapping
        norm = mcolors.Normalize(0, 1)
        cmap = cm.get_cmap("viridis")
        rgba = (cmap(norm(grid_data["prob"])) * 255).astype(int)
        
        grid_viz = grid_data.copy()
        grid_viz["r"] = rgba[:, 0]
        grid_viz["g"] = rgba[:, 1] 
        grid_viz["b"] = rgba[:, 2]
        grid_viz["a"] = 140
        
        # Create heat map layer
        heat_layer = pdk.Layer(
            "ScatterplotLayer",
            grid_viz,
            get_position="[lon, lat]",
            get_fill_color="[r, g, b, a]",
            get_radius=250,
            pickable=True
        )
        
        # Create observation points layer
        obs_layer = pdk.Layer(
            "ScatterplotLayer",
            obs_data,
            get_position="[lon, lat]",
            get_fill_color="[255, 0, 0, 200]",
            get_radius=100,
            pickable=True
        )
        
        # Map view
        view_state = pdk.ViewState(
            latitude=37.865,
            longitude=-119.538,
            zoom=9,
            pitch=0
        )
        
        # Render map
        tooltip_text = "Habitat Probability: {prob:.2f}\nElevation: {elev:.0f}m\nSlope: {slope:.1f}°"
        
        # Add bias information if available
        if 'bias_factor' in grid_data.columns:
            tooltip_text += "\nProximity Bias: {bias_factor:.2f}x"
        if 'base_prob' in grid_data.columns:
            tooltip_text += "\nBase Probability: {base_prob:.2f}"
        
        deck = pdk.Deck(
            layers=[heat_layer, obs_layer],
            initial_view_state=view_state,
            tooltip={"text": tooltip_text}
        )
        
        st.pydeck_chart(deck)
        
        # Legend
        st.caption("🟢🟡🔴 **Color Scale:** Green = High suitability, Yellow = Medium, Purple = Low")
        st.caption("🔴 **Red dots:** Known Great Gray Owl sightings from iNaturalist & GBIF")
        
        # Show proximity bias info if available
        if 'bias_factor' in grid_data.columns:
            avg_bias = grid_data['bias_factor'].mean()
            max_bias = grid_data['bias_factor'].max()
            high_bias_count = len(grid_data[grid_data['bias_factor'] > 1.5])
            st.caption(f"📍 **Proximity Bias:** {high_bias_count} locations boosted near real observations (avg: {avg_bias:.2f}x, max: {max_bias:.2f}x)")
        
        # Statistics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Grid Points", len(grid_data))
        with col_b:
            st.metric("Observations", len(obs_data))
        with col_c:
            avg_prob = grid_data["prob"].mean()
            st.metric("Avg. Suitability", f"{avg_prob:.1%}")
    else:
        st.error("No grid data available for visualization")

# Information section
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    **Model Features:**
    - **Elevation**: Higher elevations generally preferred by Great Gray Owls
    - **Slope**: Moderate slopes preferred for hunting and nesting
    - **Distance to Roads**: Owls prefer areas away from human disturbance
    - **Distance to Water**: Proximity to water bodies increases prey availability
    
    **Proximity Bias System:**
    - **Real Observations**: Areas within 5km of actual iNaturalist/GBIF sightings get boosted probability
    - **Distance Weighting**: Closer to real sightings = higher boost (up to 2x probability)
    - **Conservative Approach**: Only verified, research-grade observations used for bias
    - **Smart Filtering**: Distinguishes between real sightings and literature locations
    
    **Data Sources:**
    - **Real Owl Observations**: iNaturalist & GBIF research-grade records from Yosemite
    - **Environmental Data**: Yosemite-specific elevation, slope, and infrastructure modeling
    - **Literature Locations**: Documented Great Gray Owl sites from scientific papers
    
    **Model**: Random Forest classifier + proximity-weighted habitat suitability
    
    **Focus Area**: Yosemite National Park (37.5-38.2°N, 119.0-120.0°W)
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data: iNaturalist, SRTM, OpenStreetMap")
