"""
Great Gray Owl habitat-suitability demo for Yosemite NP
-------------------------------------------------------
Streamlit app that (1) grabs open data, (2) trains a Random-Forest,
(3) renders a probability heat-map, and (4) lets the user type lat/lon
to see the model score at that spot.

Designed for quick proof-of-concept – runs in ~2-3 min cold-start
inside the Cloud Run container.  Not optimised for large-scale use!
"""
import numpy as np, pandas as pd, requests, streamlit as st, pydeck as pdk
st.set_page_config(layout="wide")
import os, io, zipfile, warnings, functools, tempfile
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.cm as cm, matplotlib.colors as mcolors

# geopandas / rasterio / osmnx are heavy; import lazily so Streamlit's
# “Running...” spinner shows progress.
with st.spinner("Loading geospatial libraries…"):
    import geopandas as gpd, rasterio
    import osmnx as ox
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling

st.title("🦉 Great Gray Owl – habitat suitability in Yosemite NP")

###############################################################################
# Helper section – wrapped in @st.cache_resource so it runs only once per pod #
###############################################################################

@st.cache_resource(show_spinner=True, ttl=24*3600)   # keep result 24 h
def build_model_and_layers():
    st.info("Step 1/5 – Downloading owl observations from iNaturalist…")
    TAXON_ID = 19890        # Strix nebulosa
    BBOX = "37.5,-120,38.2,-119"   # (minLat,minLon,maxLat,maxLon)
    obs_url = f"https://api.inaturalist.org/v1/observations?taxon_id={TAXON_ID}&bbox={BBOX}&quality_grade=research&per_page=200"
    obs = []
    page = 1
    while True:
        resp = requests.get(obs_url + f"&page={page}", timeout=30)
        data = resp.json()
        obs.extend(data["results"])
        if len(data["results"]) < 200: break
        page += 1
    gdf_obs = gpd.GeoDataFrame(
        geometry=[Point(o["geojson"]["coordinates"][0], o["geojson"]["coordinates"][1])
                  for o in obs],
        crs="EPSG:4326")
    if gdf_obs.empty:
        st.stop()

    st.info("Step 2/5 – Fetching Yosemite NP boundary shapefile…")
    BOUND_ZIP = "https://irma.nps.gov/DataStore/DownloadFile/693229?Reference=2301294"
    zdata = requests.get(BOUND_ZIP, timeout=60).content
    with zipfile.ZipFile(io.BytesIO(zdata)) as z, tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        shp = next(p for p in z.namelist() if p.endswith(".shp"))
        gdf_bound = gpd.read_file(os.path.join(tmp, shp))
    park_poly = gdf_bound.unary_union

    st.info("Step 3/5 – Downloading + mosaicking SRTM DEM tiles…")
    tiles = ["N37W120", "N38W120", "N37W119", "N38W119"]
    srcs = []
    for t in tiles:
        url = f"https://srtm.kurviger.de/SRTM1/Region_01/{t}.hgt.zip"
        buf = requests.get(url, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(buf)) as z:
            hgt = z.namelist()[0]
            mem = rasterio.io.MemoryFile(z.read(hgt))
            srcs.append(mem.open(driver="SRTMHGT"))
    mosaic, trans = merge(srcs)
    dem = mosaic[0]          # 2-d array

    # derive slope (very rough) ------------------------------------------------
    lat_center = (park_poly.bounds[1] + park_poly.bounds[3]) / 2
    m_per_deg_lon = 111_320 * np.cos(np.deg2rad(lat_center))
    m_per_deg_lat = 111_320
    pix_deg = abs(trans.a)   # ~0.000278°
    dz_dlat = np.gradient(dem, axis=0) / (pix_deg*m_per_deg_lat)
    dz_dlon = np.gradient(dem, axis=1) / (pix_deg*m_per_deg_lon)
    slope = np.degrees(np.arctan(np.hypot(dz_dlat, dz_dlon)))

    st.info("Step 4/5 – Fetching OpenStreetMap water & road layers…")
    n, s, e, w = park_poly.bounds[3]+.02, park_poly.bounds[1]-.02, park_poly.bounds[2]+.02, park_poly.bounds[0]-.02
    water = ox.geometries_from_bbox(n, s, e, w, tags={"natural":["water"], "waterway":True})
    roads = ox.geometries_from_bbox(n, s, e, w, tags={"highway":True})
    if "highway" in roads.columns:
        roads = roads[~roads["highway"].isin(["path","footway","cycleway"])]

    EPSG_UTM = "EPSG:32611"   # UTM zone 11N
    water_u = gpd.GeoSeries([water.unary_union], crs="EPSG:4326").to_crs(EPSG_UTM).iloc[0]
    road_u  = gpd.GeoSeries([roads.unary_union], crs="EPSG:4326").to_crs(EPSG_UTM).iloc[0]
    obs_u   = gdf_obs.to_crs(EPSG_UTM)

    st.info("Step 5/5 – Building training set & Random-Forest…")
    tf_fwd  = Transformer.from_crs("EPSG:4326", EPSG_UTM, always_xy=True)
    tf_back = Transformer.from_crs(EPSG_UTM, "EPSG:4326", always_xy=True)

    def sample_feats(lat, lon):
        # DEM / slope index
        c = int((lon - trans.c) / trans.a)
        r = int((lat - trans.f) / trans.e)
        c = np.clip(c, 0, dem.shape[1]-1)
        r = np.clip(r, 0, dem.shape[0]-1)
        elev  = float(dem[r, c])
        slp   = float(slope[r, c])
        x,y   = tf_fwd.transform(lon, lat)
        p_u   = Point(x, y)
        return elev, slp, p_u.distance(road_u), p_u.distance(water_u)

    pres_feats = [sample_feats(p.y, p.x) for p in gdf_obs.geometry]
    df_pres = pd.DataFrame(pres_feats, columns=["elev","slope","d_road","d_water"]); df_pres["label"]=1

    rng = np.random.default_rng(0)
    park_u = gpd.GeoSeries([park_poly], crs="EPSG:4326").to_crs(EPSG_UTM).iloc[0]
    minx,maxx,miny,maxy = *park_u.bounds[:2],*park_u.bounds[2:]
    bg = []
    while len(bg) < 1000:
        x = rng.uniform(minx, maxx); y = rng.uniform(miny, maxy)
        pt = Point(x,y)
        if not park_u.contains(pt): continue
        if obs_u.distance(pt).min() < 500: continue
        lon,lat = tf_back.transform(x,y)
        bg.append(sample_feats(lat,lon))
    df_bg = pd.DataFrame(bg, columns=["elev","slope","d_road","d_water"]); df_bg["label"]=0

    df_train = pd.concat([df_pres, df_bg], ignore_index=True)
    X = df_train[["elev","slope","d_road","d_water"]].values
    y = df_train["label"].values
    rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    rf.fit(X,y)

    # Pre-compute 500 m grid predictions for heat-map --------------------------
    xs = np.arange(minx, maxx, 500)
    ys = np.arange(miny, maxy, 500)
    grid_lon,grid_lat,grid_prob = [],[],[]
    for x in xs:
        for y in ys:
            pt=Point(x,y)
            if not park_u.contains(pt): continue
            lon,lat = tf_back.transform(x,y)
            elev,slp,dr,dw=sample_feats(lat,lon)
            pr = rf.predict_proba([[elev,slp,dr,dw]])[0,1]
            grid_lon.append(lon); grid_lat.append(lat); grid_prob.append(pr)

    return {
        "rf": rf,
        "sample_feats": sample_feats,
        "grid": pd.DataFrame({"lon":grid_lon,"lat":grid_lat,"prob":grid_prob}),
        "obs": gdf_obs
    }

data = build_model_and_layers()
rf, sample_feats = data["rf"], data["sample_feats"]

###############################
#  UI – left column = controls #
###############################
col1, col2 = st.columns([1,2], gap="large")

with col1:
    st.subheader("Try your own location")
    lat_in = st.number_input("Latitude", value=37.85, format="%.6f")
    lon_in = st.number_input("Longitude", value=-119.55, format="%.6f")
    if st.button("Predict probability"):
        elev,slp,dr,dw = sample_feats(lat_in, lon_in)
        prob = rf.predict_proba([[elev,slp,dr,dw]])[0,1]
        st.success(f"Predicted owl-presence probability: **{prob:.2%}**")
        st.caption(f"Elevation {elev:.0f} m – Slope {slp:.1f}° – {dr/1000:.2f} km to road – {dw/1000:.2f} km to water")

with col2:
    st.subheader("Model heat-map")
    # Colour points by prob
    norm = mcolors.Normalize(0,1); cmap = cm.get_cmap("viridis")
    data["grid"]["r"] = (cmap(norm(data["grid"]["prob"]))[:,0]*255).astype(int)
    data["grid"]["g"] = (cmap(norm(data["grid"]["prob"]))[:,1]*255).astype(int)
    data["grid"]["b"] = (cmap(norm(data["grid"]["prob"]))[:,2]*255).astype(int)
    data["grid"]["a"] = 140
    layer_heat = pdk.Layer(
        "ScatterplotLayer",
        data["grid"],
        get_position='[lon,lat]',
        get_fill_color='[r,g,b,a]',
        get_radius=250, pickable=False)
    layer_obs = pdk.Layer(
        "ScatterplotLayer",
        data["obs"],
        get_position='[geometry.x,geometry.y]',
        get_fill_color='[255,0,0,200]',
        get_radius=100, pickable=True)
    view_state = pdk.ViewState(latitude=37.865, longitude=-119.538, zoom=9)
    st.pydeck_chart(pdk.Deck(layers=[layer_heat, layer_obs], initial_view_state=view_state))
    st.caption("Yellow = higher predicted suitability.  Red dots = known research-grade iNaturalist sightings.")
