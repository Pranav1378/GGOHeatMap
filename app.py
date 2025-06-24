"""
Instant Streamlit viewer for Great-Gray-Owl habitat map (Yosemite NP)
Loads pre-computed model.pkl, grid.csv, obs.csv baked in the Docker image.
"""

import numpy as np, pandas as pd, streamlit as st, pydeck as pdk
import joblib, matplotlib.cm as cm, matplotlib.colors as mcolors
from shapely.geometry import Point
import geopandas as gpd

st.set_page_config(layout="wide")
st.title("🦉 Great Gray Owl – habitat suitability in Yosemite NP")

# ── load baked artefacts ───────────────────────────────────────────
rf       = joblib.load("model.pkl")
grid_df  = pd.read_csv("grid.csv")
obs_df   = pd.read_csv("obs.csv")
gdf_obs  = gpd.GeoDataFrame(
    obs_df,
    geometry=[Point(x, y) for x, y in zip(obs_df.lon, obs_df.lat)],
    crs="EPSG:4326",
)

# helper for point prediction
def fvec(lat, lon, tf_cache={}):
    from pyproj import Transformer
    if "fwd" not in tf_cache:
        tf_cache["fwd"]  = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    if "back" not in tf_cache:
        tf_cache["back"] = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
    # quick nearest-grid lookup so we don't ship DEM & slope arrays:
    row = (np.abs(grid_df.lat - lat)).idxmin()
    elev, slp, dr, dw = (
        grid_df.loc[row, ["elev","slope","d_road","d_water"]]
        if {"elev","slope","d_road","d_water"}.issubset(grid_df.columns)
        else (0,0,1e3,1e3)      # fallback if trimmed grid
    )
    return elev, slp, dr, dw

# ── UI ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Try a coordinate")
    lat = st.number_input("Latitude", 37.85, format="%.6f")
    lon = st.number_input("Longitude", -119.55, format="%.6f")
    if st.button("Predict"):
        prob = rf.predict_proba([fvec(lat, lon)])[0, 1]
        st.success(f"Owl-presence probability **{prob:.1%}**")

with col2:
    st.subheader("Habitat heat-map")
    norm, cmap = mcolors.Normalize(0, 1), cm.get_cmap("viridis")
    rgba = (cmap(norm(grid_df.prob)) * 255).astype(int)
    g = grid_df.copy()
    g[["r", "g", "b"]] = rgba[:, :3]
    g["a"] = 140
    layer_heat = pdk.Layer(
        "ScatterplotLayer", g,
        get_position='[lon,lat]', get_fill_color='[r,g,b,a]', get_radius=250
    )
    layer_obs  = pdk.Layer(
        "ScatterplotLayer", gdf_obs,
        get_position='[geometry.x,geometry.y]', get_fill_color='[255,0,0,200]',
        get_radius=100
    )
    view = pdk.ViewState(latitude=37.865, longitude=-119.538, zoom=9)
    st.pydeck_chart(pdk.Deck(layers=[layer_heat, layer_obs], initial_view_state=view))
    st.caption("Yellow = higher suitability. Red dots = iNaturalist records.")
