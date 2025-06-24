import streamlit as st, pandas as pd, joblib, pydeck as pdk
import matplotlib.cm as cm, matplotlib.colors as mcolors
import numpy as np

st.set_page_config(layout="wide")
st.title("🦉 Great Gray Owl – habitat suitability in Yosemite NP")

# ---------- load pre-baked artifacts ----------
@st.cache_resource
def load():
    rf   = joblib.load("model.pkl")
    grid = pd.read_csv("grid.csv")
    obs  = pd.read_csv("obs.csv")          # lon / lat columns
    return rf, grid, obs
rf, grid, obs = load()

# ---------- UI ----------
col1, col2 = st.columns([1,2], gap="large")

with col1:
    st.subheader("Try your own location")
    lat = st.number_input("Latitude", 37.85, format="%.6f")
    lon = st.number_input("Longitude", -119.55, format="%.6f")
    if st.button("Predict"):
        # nearest-grid fallback (fast & good enough at 500 m resolution)
        idx  = ((grid.lat-lat)**2 + (grid.lon-lon)**2).idxmin()
        prob = grid.prob.iloc[idx]
        st.success(f"Predicted owl-presence probability: **{prob:.1%}**")
        st.caption("(Nearest 500 m grid cell)")

with col2:
    st.subheader("Habitat-suitability heat-map")
    norm, cmap = mcolors.Normalize(0,1), cm.get_cmap("viridis")
    rgba = (cmap(norm(grid.prob))*255).astype(int)
    grid[["r","g","b"]] = rgba[:,:3]
    grid["a"] = 140

    layer_heat = pdk.Layer(
        "ScatterplotLayer", grid,
        get_position='[lon,lat]', get_fill_color='[r,g,b,a]', get_radius=250
    )
    layer_obs = pdk.Layer(
        "ScatterplotLayer", obs,
        get_position='[lon,lat]', get_fill_color='[255,0,0,200]', get_radius=100
    )
    view = pdk.ViewState(latitude=37.865, longitude=-119.538, zoom=9)
    st.pydeck_chart(pdk.Deck(layers=[layer_heat, layer_obs], initial_view_state=view))
    st.caption("Yellow = higher suitability • Red dots = known iNaturalist sightings")
