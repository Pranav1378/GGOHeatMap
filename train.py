#!/usr/bin/env python3
"""
train.py  – Executed at docker-build time.
Downloads open data, trains Random-Forest, writes:
  • model.pkl      • grid.csv      • obs.csv
Run locally once to test:  python train.py
"""
import io, zipfile, requests, joblib, warnings, json, tempfile
import numpy as np, pandas as pd, geopandas as gpd, rasterio
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
BASE  = f"https://api.inaturalist.org/v1/observations?taxon_id={TID}&bbox={BBOX}&quality_grade=research&per_page=200"
obs, page = [], 1
while True:
    r = requests.get(f"{BASE}&page={page}", timeout=30).json()
    obs += r["results"];  page += 1
    if len(r["results"]) < 200: break
coords = [o["geojson"]["coordinates"] for o in obs if o.get("geojson")]
lons, lats = zip(*coords)
gdf_obs = gpd.GeoDataFrame(
    {"lon":lons,"lat":lats},
    geometry=[Point(x,y) for x,y in coords],
    crs="EPSG:4326")

# 2) DEM + slope ---------------------------------------------------
print("▸ assembling SRTM mosaic…")
tiles  = ["N37W120","N38W120","N37W119","N38W119"]
srcs   = []
for t in tiles:
    z = requests.get(f"https://srtm.kurviger.de/SRTM1/Region_01/{t}.hgt.zip",timeout=60).content
    if not z.startswith(b"PK"): continue
    with zipfile.ZipFile(io.BytesIO(z)) as zz:
        hgt = zz.namelist()[0]
        data = zz.read(hgt)
        # Use MemoryFile instead of temporary file
        mem = rasterio.io.MemoryFile(data)
        srcs.append(mem.open(driver="SRTMHGT"))
if not srcs: raise RuntimeError("No DEM tiles")
mosaic, trans = merge(srcs); dem = mosaic[0]
lat_c = 37.9
mpx, mpy = 111_320*np.cos(np.deg2rad(lat_c)), 111_320
deg = abs(trans.a)
slope = np.degrees(np.arctan(
        np.hypot(np.gradient(dem,axis=0)/(deg*mpy),
                 np.gradient(dem,axis=1)/(deg*mpx))
))

# 3) simple distances to water / road via OSM ----------------------
print("▸ fetching OSM layers…")
import osmnx as ox
north,south,east,west = max(lats)+.02,min(lats)-.02,max(lons)+.02,min(lons)-.02
water = ox.geometries_from_bbox(north,south,east,west, {"natural":["water"],"waterway":True})
roads = ox.geometries_from_bbox(north,south,east,west, {"highway":True})
if "highway" in roads: roads = roads[~roads.highway.isin(["path","footway","cycleway"])]

UTM = "EPSG:32611"
water_u = gpd.GeoSeries([water.unary_union], crs="EPSG:4326").to_crs(UTM).iloc[0]
roads_u = gpd.GeoSeries([roads.unary_union], crs="EPSG:4326").to_crs(UTM).iloc[0]
obs_u   = gdf_obs.to_crs(UTM)
tf_fwd  = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
tf_back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)

def feats(lat, lon):
    col = int((lon - trans.c)/trans.a);  row = int((lat - trans.f)/trans.e)
    col = np.clip(col,0,dem.shape[1]-1); row = np.clip(row,0,dem.shape[0]-1)
    elev, slp = float(dem[row,col]), float(slope[row,col])
    x,y = tf_fwd.transform(lon,lat);  p = Point(x,y)
    return elev, slp, p.distance(roads_u), p.distance(water_u)

print("▸ building train set…")
pres = [feats(y,x) for x,y in coords]
dfp  = pd.DataFrame(pres, columns=["elev","slope","d_road","d_water"]); dfp["label"]=1
park  = box(west,south,east,north)
park_u= gpd.GeoSeries([park], crs="EPSG:4326").to_crs(UTM).iloc[0]
rng   = np.random.default_rng(0);  bg=[]
while len(bg)<1000:
    x = rng.uniform(*park_u.bounds[0::2]); y = rng.uniform(*park_u.bounds[1::2])
    p = Point(x,y)
    if not park_u.contains(p): continue
    if obs_u.distance(p).min()<500: continue
    lon,lat = tf_back.transform(x,y)
    bg.append(feats(lat,lon))
dfb = pd.DataFrame(bg,columns=dfp.columns); dfb["label"]=0
df  = pd.concat([dfp,dfb]); X,y = df.drop("label",axis=1), df.label

print("▸ fitting Random-Forest…")
rf = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                            random_state=42, n_jobs=-1)
rf.fit(X, y)

# 4) grid prediction (500 m) ---------------------------------------
print("▸ predicting 500 m grid…")
minx,maxx,miny,maxy = park_u.bounds
G=[]
for x in np.arange(minx,maxx,500):
    for y in np.arange(miny,maxy,500):
        if not park_u.contains(Point(x,y)): continue
        lon,lat = tf_back.transform(x,y)
        G.append((lon,lat,*feats(lat,lon)))
grid = pd.DataFrame(G, columns=["lon","lat","elev","slope","d_road","d_water"])
grid["prob"] = rf.predict_proba(grid[["elev","slope","d_road","d_water"]])[:,1]

# 5) save artefacts -------------------------------------------------
joblib.dump(rf, "model.pkl")
grid[["lon","lat","prob","elev","slope","d_road","d_water"]].to_csv("grid.csv", index=False)
gdf_obs[["lon","lat"]].to_csv("obs.csv", index=False)
print("✓ model.pkl, grid.csv, obs.csv saved")
