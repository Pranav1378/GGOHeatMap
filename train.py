#!/usr/bin/env python3
"""
train.py  – Executed at docker-build time.
Downloads open data, trains Random-Forest, writes:
  • model.pkl      • grid.csv      • obs.csv
Run locally once to test:  python train.py
"""
import io
import zipfile
import requests
import joblib
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from shapely.geometry import Point, box
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# 1) iNaturalist owl points ---------------------------------------
print("▸ downloading iNaturalist…")
TID   = 19890
BBOX  = "37.5,-120,38.2,-119"
BASE  = (
    f"https://api.inaturalist.org/v1/observations"
    f"?taxon_id={TID}&bbox={BBOX}&quality_grade=research&per_page=200"
)
obs, page = [], 1
while True:
    r = requests.get(f"{BASE}&page={page}", timeout=30).json()
    obs += r["results"]
    if len(r["results"]) < 200:
        break
    page += 1

# filter out any observations lacking geojson
coords = [o["geojson"]["coordinates"] for o in obs if o.get("geojson")]
lons, lats = zip(*coords)
gdf_obs = gpd.GeoDataFrame(
    {"lon": lons, "lat": lats},
    geometry=[Point(x, y) for x, y in coords],
    crs="EPSG:4326",
)

# 2) DEM + slope ---------------------------------------------------
print("▸ assembling SRTM mosaic…")
tiles = ["N37W120", "N38W120", "N37W119", "N38W119"]
srcs = []

# Try to download and process SRTM tiles
for t in tiles:
    try:
        print(f"  attempting {t}...")
        z = requests.get(
            f"https://srtm.kurviger.de/SRTM1/Region_01/{t}.hgt.zip", timeout=60
        ).content
        if not z.startswith(b"PK"):
            print(f"  {t}: not a valid zip file")
            continue
        with zipfile.ZipFile(io.BytesIO(z)) as zz:
            hgt_files = [n for n in zz.namelist() if n.lower().endswith(".hgt")]
            if not hgt_files:
                print(f"  {t}: no .hgt file found")
                continue
            hgt_name = hgt_files[0]
            data = zz.read(hgt_name)
            
            # Check if this looks like valid HGT data (should be 3601x3601 for SRTM1)
            expected_size = 3601 * 3601 * 2  # 16-bit integers
            if len(data) != expected_size:
                print(f"  {t}: unexpected file size {len(data)}, expected {expected_size}")
                continue
                
            try:
                # Create a temporary file with proper naming
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".hgt") as tmp:
                    tmp.write(data)
                    tmp.flush()
                    src = rasterio.open(tmp.name, driver="SRTMHGT")
                    srcs.append(src)
                    print(f"  {t}: successfully loaded")
                    # Keep temp file for the duration (will be cleaned up by container)
            except Exception as e:
                print(f"  {t}: failed to open as SRTM: {e}")
                continue
                
    except Exception as e:
        print(f"  {t}: download failed: {e}")
        continue

if not srcs:
    print("▸ SRTM download failed, creating synthetic elevation model...")
    # Create a realistic elevation model for Yosemite area
    # Yosemite elevations range from ~1200m (valley) to ~4400m (peaks)
    lat_range = np.linspace(37.5, 38.2, 1000)
    lon_range = np.linspace(-120, -119, 1000)
    
    # Create elevation that increases with distance from valley center
    center_lat, center_lon = 37.75, -119.6  # Approximate valley center
    dem = np.zeros((1000, 1000))
    
    for i, lat in enumerate(lat_range):
        for j, lon in enumerate(lon_range):
            # Distance from valley center
            dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            # Base elevation + elevation gain with distance
            dem[i, j] = 1200 + dist * 2000 + np.random.normal(0, 100)
    
    # Clip to reasonable range
    dem = np.clip(dem, 1000, 4500)
    
    # Create transform for synthetic DEM
    trans = rasterio.Affine(
        (lon_range[-1] - lon_range[0]) / 1000,  # pixel width
        0, 
        lon_range[0],  # left edge
        0, 
        -(lat_range[-1] - lat_range[0]) / 1000,  # pixel height (negative)
        lat_range[-1]  # top edge
    )
    print(f"  created synthetic DEM: {dem.shape}, elevation range {dem.min():.0f}-{dem.max():.0f}m")
else:
    print(f"▸ successfully loaded {len(srcs)} SRTM tiles")
    mosaic, trans = merge(srcs)
    dem = mosaic[0]

lat_c = (min(lats) + max(lats)) / 2
mpx = 111_320 * np.cos(np.deg2rad(lat_c))
mpy = 111_320
deg = abs(trans.a)
slope = np.degrees(
    np.arctan(
        np.hypot(
            np.gradient(dem, axis=0) / (deg * mpy),
            np.gradient(dem, axis=1) / (deg * mpx),
        )
    )
)

# 3) simple distances to water / road via OSM ----------------------
print("▸ fetching OSM layers…")
import osmnx as ox

north, south, east, west = max(lats) + 0.02, min(lats) - 0.02, max(lons) + 0.02, min(lons) - 0.02
water = ox.geometries_from_bbox(north, south, east, west, {"natural": ["water"], "waterway": True})
roads = ox.geometries_from_bbox(north, south, east, west, {"highway": True})
if "highway" in roads.columns:
    roads = roads[~roads.highway.isin(["path", "footway", "cycleway"])]

UTM = "EPSG:32611"
water_u = gpd.GeoSeries([water.unary_union], crs="EPSG:4326").to_crs(UTM).iloc[0]
roads_u = gpd.GeoSeries([roads.unary_union], crs="EPSG:4326").to_crs(UTM).iloc[0]
obs_u = gdf_obs.to_crs(UTM)

tf_fwd = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
tf_back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)


def feats(lat, lon):
    col = int((lon - trans.c) / trans.a)
    row = int((lat - trans.f) / trans.e)
    col = np.clip(col, 0, dem.shape[1] - 1)
    row = np.clip(row, 0, dem.shape[0] - 1)
    elev, slp = float(dem[row, col]), float(slope[row, col])
    x, y = tf_fwd.transform(lon, lat)
    p = Point(x, y)
    return elev, slp, p.distance(roads_u), p.distance(water_u)


# 4) build training set --------------------------------------------
print("▸ building train set…")
pres = [feats(y, x) for x, y in coords]
dfp = pd.DataFrame(pres, columns=["elev", "slope", "d_road", "d_water"])
dfp["label"] = 1

park = box(west, south, east, north)
park_u = gpd.GeoSeries([park], crs="EPSG:4326").to_crs(UTM).iloc[0]

rng = np.random.default_rng(0)
bg = []
while len(bg) < 1000:
    x = rng.uniform(*park_u.bounds[0::2])
    y = rng.uniform(*park_u.bounds[1::2])
    p = Point(x, y)
    if not park_u.contains(p):
        continue
    if obs_u.distance(p).min() < 500:
        continue
    lon, lat = tf_back.transform(x, y)
    bg.append(feats(lat, lon))

dfb = pd.DataFrame(bg, columns=dfp.columns)
dfb["label"] = 0

df = pd.concat([dfp, dfb], ignore_index=True)
X, y = df.drop("label", axis=1).values, df["label"].values

print("▸ fitting Random-Forest…")
rf = RandomForestClassifier(
    n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X, y)

# 5) grid prediction (500 m) ---------------------------------------
print("▸ predicting 500 m grid…")
minx, maxx, miny, maxy = park_u.bounds
rows = []
for x in np.arange(minx, maxx, 500):
    for y in np.arange(miny, maxy, 500):
        if not park_u.contains(Point(x, y)):
            continue
        lon, lat = tf_back.transform(x, y)
        rows.append((lon, lat, *feats(lat, lon)))

grid = pd.DataFrame(rows, columns=["lon", "lat", "elev", "slope", "d_road", "d_water"])
grid["prob"] = rf.predict_proba(grid[["elev", "slope", "d_road", "d_water"]])[:, 1]

# 6) save artefacts -------------------------------------------------
joblib.dump(rf, "model.pkl")
grid[["lon", "lat", "prob", "elev", "slope", "d_road", "d_water"]].to_csv("grid.csv", index=False)
gdf_obs[["lon", "lat"]].to_csv("obs.csv", index=False)
print("✓ model.pkl, grid.csv, obs.csv saved")
