"""
Great Gray Owl habitat-suitability demo for Yosemite NP
-------------------------------------------------------
Streamlit app that (1) grabs open data, (2) trains a Random-Forest,
(3) renders a probability heat-map, and (4) lets the user type lat/lon
to see the model score at that spot.

Cold start on a good network ≈2–3 min (everything happens at runtime).
"""
import numpy as np, pandas as pd, requests, streamlit as st, pydeck as pdk
st.set_page_config(layout="wide")
import os, io, zipfile, warnings, tempfile
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.cm as cm, matplotlib.colors as mcolors

# heavy GIS libs loaded lazily so Streamlit shows progress
with st.spinner("Loading geospatial libraries…"):
    import geopandas as gpd, rasterio
    import osmnx as ox
    from rasterio.merge import merge

st.title("🦉 Great Gray Owl – habitat suitability in Yosemite NP")

###############################################################################
# Helper section – cached so it runs once per container                       #
###############################################################################
@st.cache_resource(show_spinner=True, ttl=24*3600)      # reuse for a day
def build_model_and_layers():
    st.info("Step 1/5 – Downloading owl observations from iNaturalist…")
    TAXON_ID = 19890
    BBOX = "37.5,-120,38.2,-119"
    obs_url = (
        f"https://api.inaturalist.org/v1/observations?"
        f"taxon_id={TAXON_ID}&bbox={BBOX}&quality_grade=research&per_page=200"
    )
    obs, page = [], 1
    while True:
        r = requests.get(obs_url + f"&page={page}", timeout=30).json()
        obs += r["results"]
        if len(r["results"]) < 200: break
        page += 1
    valid = [o for o in obs if o.get("geojson") and o["geojson"].get("coordinates")]
    gdf_obs = gpd.GeoDataFrame(
        geometry=[Point(c) for c in (o["geojson"]["coordinates"] for o in valid)],
        crs="EPSG:4326"
    )

    st.info("Step 2/5 – Building Yosemite bounding polygon…")
    lat_min, lon_min, lat_max, lon_max = 37.5, -120, 38.2, -119
    from shapely.geometry import box
    park_poly = box(lon_min, lat_min, lon_max, lat_max)

    st.info("Step 3/5 – Downloading and mosaicking SRTM DEM tiles…")
    tiles = ["N37W120", "N38W120", "N37W119", "N38W119"]
    srcs  = []
    for t in tiles:
        buf = requests.get(
            f"https://srtm.kurviger.de/SRTM1/Region_01/{t}.hgt.zip",
            timeout=60
        ).content
        with zipfile.ZipFile(io.BytesIO(buf)) as z:
            h = z.namelist()[0]
            mem = rasterio.io.MemoryFile(z.read(h))
            srcs.append(mem.open(driver="SRTMHGT"))
    mosaic, trans = merge(srcs)
    dem = mosaic[0]

    lat_c = (lat_min + lat_max)/2
    mpx = 111_320*np.cos(np.deg2rad(lat_c)); mpy = 111_320
    deg = abs(trans.a)
    dzdy = np.gradient(dem,axis=0)/(deg*mpy)
    dzdx = np.gradient(dem,axis=1)/(deg*mpx)
    slope = np.degrees(np.arctan(np.hypot(dzdy,dzdx)))

    st.info("Step 4/5 – Fetching OSM water & road layers…")
    n,s,e,w = lat_max+.02, lat_min-.02, lon_max+.02, lon_min-.02
    water = ox.geometries_from_bbox(n,s,e,w, {"natural":["water"],"waterway":True})
    roads = ox.geometries_from_bbox(n,s,e,w, {"highway":True})
    if "highway" in roads.columns:
        roads = roads[~roads.highway.isin(["path","footway","cycleway"])]

    EPSG_UTM = "EPSG:32611"
    water_u = gpd.GeoSeries([water.unary_union], crs="EPSG:4326").to_crs(EPSG_UTM).iloc[0]
    road_u  = gpd.GeoSeries([roads.unary_union], crs="EPSG:4326").to_crs(EPSG_UTM).iloc[0]
    obs_u   = gdf_obs.to_crs(EPSG_UTM)

    st.info("Step 5/5 – Training Random-Forest and building heat-map grid…")
    tf_fwd  = Transformer.from_crs("EPSG:4326", EPSG_UTM, always_xy=True)
    tf_back = Transformer.from_crs(EPSG_UTM, "EPSG:4326", always_xy=True)

    def fvec(lat, lon):
        c = int((lon - trans.c) / trans.a)
        r = int((lat - trans.f) / trans.e)
        c = np.clip(c,0,dem.shape[1]-1); r = np.clip(r,0,dem.shape[0]-1)
        elev = float(dem[r,c]); slp = float(slope[r,c])
        x,y  = tf_fwd.transform(lon,lat)
        pt   = Point(x,y)
        return elev, slp, pt.distance(road_u), pt.distance(water_u)

    # presence & pseudo-absence
    pres = [fvec(p.y,p.x) for p in gdf_obs.geometry]
    dfp  = pd.DataFrame(pres, columns=["elev","slope","d_road","d_water"]); dfp["label"]=1
    rng  = np.random.default_rng(0)
    park_u = gpd.GeoSeries([park_poly], crs="EPSG:4326").to_crs(EPSG_UTM).iloc[0]
    minx,maxx,miny,maxy = park_u.bounds
    bg=[]
    while len(bg)<1000:
        x = rng.uniform(minx,maxx); y = rng.uniform(miny,maxy)
        pt = Point(x,y)
        if not park_u.contains(pt): continue
        if obs_u.distance(pt).min()<500: continue
        lon,lat = tf_back.transform(x,y)
        bg.append(fvec(lat,lon))
    dfb = pd.DataFrame(bg,columns=dfp.columns); dfb["label"]=0
    df  = pd.concat([dfp,dfb],ignore_index=True)
    X,y = df.drop("label",axis=1).values, df.label.values
    rf  = RandomForestClassifier(n_estimators=150,class_weight="balanced",random_state=42)
    rf.fit(X,y)

    # 500 m grid
    xs = np.arange(minx,maxx,500); ys = np.arange(miny,maxy,500)
    glon,glat,gprob=[],[],[]
    for x in xs:
        for y in ys:
            if not park_u.contains(Point(x,y)): continue
            lon,lat = tf_back.transform(x,y)
            glon.append(lon); glat.append(lat)
            gprob.append(rf.predict_proba([fvec(lat,lon)])[0,1])
    grid = pd.DataFrame({"lon":glon,"lat":glat,"prob":gprob})

    return {"rf":rf, "fvec":fvec, "grid":grid, "obs":gdf_obs}

data = build_model_and_layers()
rf, fvec = data["rf"], data["fvec"]

############################
#  UI   #
############################
col1,col2 = st.columns([1,2], gap="large")

with col1:
    st.subheader("Try your own location")
    lat = st.number_input("Latitude", 37.85, format="%.6f")
    lon = st.number_input("Longitude", -119.55, format="%.6f")
    if st.button("Predict probability"):
        elev, slp, dr, dw = fvec(lat, lon)
        prob = rf.predict_proba([[elev, slp, dr, dw]])[0,1]
        st.success(f"Predicted owl-presence probability: **{prob:.1%}**")
        st.caption(f"Elevation {elev:.0f} m • Slope {slp:.1f}° • "
                   f"{dr/1000:.2f} km to road • {dw/1000:.2f} km to water")

with col2:
    st.subheader("Model heat-map")
    norm, cmap = mcolors.Normalize(0,1), cm.get_cmap("viridis")
    rgba = (cmap(norm(data["grid"].prob))*255).astype(int)
    g = data["grid"].copy()
    g[["r","g","b"]] = rgba[:,:3]; g["a"] = 140
    layer_heat = pdk.Layer(
        "ScatterplotLayer", g,
        get_position='[lon,lat]', get_fill_color='[r,g,b,a]', get_radius=250
    )
    layer_obs = pdk.Layer(
        "ScatterplotLayer", data["obs"],
        get_position='[geometry.x,geometry.y]', get_fill_color='[255,0,0,200]',
        get_radius=100
    )
    view = pdk.ViewState(latitude=37.865, longitude=-119.538, zoom=9)
    st.pydeck_chart(pdk.Deck(layers=[layer_heat,layer_obs], initial_view_state=view))
    st.caption("Yellow = higher suitability • Red dots = iNaturalist sightings")
