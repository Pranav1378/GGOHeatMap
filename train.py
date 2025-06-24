"""
train.py
--------
• Downloads Great-Gray-Owl observations (iNaturalist)
• Defines Yosemite NP as the BBOX polygon
• Downloads SRTM DEM & computes slope
• Fetches OSM water + roads
• Trains a Random-Forest (presence vs pseudo-absence)
• Pre-computes a 500 m grid of presence probabilities
• Saves model.pkl, grid.csv, obs.csv into /app
"""
import io, zipfile, requests, joblib
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.merge import merge
from shapely.geometry import Point, box
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Owl observations (iNaturalist)
TAXON_ID = 19890
BBOX     = "37.5,-120,38.2,-119"
base_url = (
    f"https://api.inaturalist.org/v1/observations"
    f"?taxon_id={TAXON_ID}&bbox={BBOX}"
    "&quality_grade=research&per_page=200"
)
obs, page = [], 1
while True:
    resp = requests.get(base_url + f"&page={page}", timeout=30).json()
    obs.extend(resp["results"])
    if len(resp["results"]) < 200:
        break
    page += 1

# filter out any None geojson
valid = [o for o in obs if o.get("geojson") and o["geojson"].get("coordinates")]
gdf_obs = gpd.GeoDataFrame(
    {
      "lon": [c[0] for c in (o["geojson"]["coordinates"] for o in valid)],
      "lat": [c[1] for c in (o["geojson"]["coordinates"] for o in valid)]
    },
    geometry=[Point(o["geojson"]["coordinates"]) for o in valid],
    crs="EPSG:4326"
)

# 2️⃣ Define the park polygon as your BBOX
lat_min, lon_min, lat_max, lon_max = 37.5, -120, 38.2, -119
park_poly = box(lon_min, lat_min, lon_max, lat_max)

# 3️⃣ Download + mosaic SRTM DEM tiles
tiles = ["N37W120","N38W120","N37W119","N38W119"]
srcs  = []
for t in tiles:
    buf = requests.get(
        f"https://srtm.kurviger.de/SRTM1/Region_01/{t}.hgt.zip",
        timeout=60
    ).content
    with zipfile.ZipFile(io.BytesIO(buf)) as zz:
        h = zz.namelist()[0]
        mem = rasterio.io.MemoryFile(zz.read(h))
        srcs.append(mem.open(driver="SRTMHGT"))
mosaic, trans = merge(srcs)
dem = mosaic[0]

# compute slope (approximate)
latc = (lat_min + lat_max) / 2
mpx  = 111_320 * np.cos(np.deg2rad(latc))
mpy  = 111_320
deg  = abs(trans.a)
dzdy = np.gradient(dem, axis=0) / (deg*mpy)
dzdx = np.gradient(dem, axis=1) / (deg*mpx)
slope = np.degrees(np.arctan(np.hypot(dzdy, dzdx)))

# 4️⃣ OSM water & roads
import osmnx as ox
n,s,e,w = lat_max+0.02, lat_min-0.02, lon_max+0.02, lon_min-0.02
water = ox.geometries_from_bbox(n, s, e, w,
    tags={"natural":["water"],"waterway":True}
)
roads = ox.geometries_from_bbox(n, s, e, w, tags={"highway":True})
# drop paths/footways
if "highway" in roads.columns:
    roads = roads[~roads.highway.isin(["path","footway","cycleway"])]

# reproject to UTM for distances
UTM = "EPSG:32611"
water_u = gpd.GeoSeries([water.unary_union], crs="EPSG:4326")\
           .to_crs(UTM).iloc[0]
road_u  = gpd.GeoSeries([roads.unary_union], crs="EPSG:4326")\
           .to_crs(UTM).iloc[0]
obs_u   = gdf_obs.to_crs(UTM)
park_u  = gpd.GeoSeries([park_poly], crs="EPSG:4326")\
           .to_crs(UTM).iloc[0]

tf_fwd  = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
tf_back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)

def feats(lat, lon):
    # sample DEM & slope
    c = int((lon - trans.c) / trans.a)
    r = int((lat - trans.f) / trans.e)
    c,r = np.clip(c,0,dem.shape[1]-1), np.clip(r,0,dem.shape[0]-1)
    elev = float(dem[r,c]); slp = float(slope[r,c])
    # distance to road/water in UTM
    x,y = tf_fwd.transform(lon, lat)
    p   = Point(x,y)
    return [elev, slp, p.distance(road_u), p.distance(water_u)]

# 5️⃣ Build presence & pseudo-absence
pres = [feats(p.y,p.x) for p in gdf_obs.geometry]
dfp  = pd.DataFrame(pres,columns=["elev","slope","d_road","d_water"])
dfp["label"] = 1

rng = np.random.default_rng(0)
minx,maxx,miny,maxy = park_u.bounds
bg=[]
while len(bg) < 1000:
    x,y = rng.uniform(minx,maxx), rng.uniform(miny,maxy)
    p   = Point(x,y)
    if not park_u.contains(p): continue
    if obs_u.distance(p).min() < 500: continue
    lon,lat = tf_back.transform(x,y)
    bg.append(feats(lat,lon))
dfb = pd.DataFrame(bg,columns=dfp.columns); dfb["label"]=0

df = pd.concat([dfp,dfb],ignore_index=True)
X, y = df[["elev","slope","d_road","d_water"]].values, df.label.values
rf   = RandomForestClassifier(n_estimators=150,class_weight="balanced",random_state=42)
rf.fit(X,y)

# 6️⃣ Precompute 500 m grid probabilities
xs = np.arange(minx,maxx,500)
ys = np.arange(miny,maxy,500)
lons,lats,probs = [],[],[]
for x in xs:
    for y in ys:
        pt = Point(x,y)
        if not park_u.contains(pt): continue
        lon,lat = tf_back.transform(x,y)
        pr = rf.predict_proba([feats(lat,lon)])[0,1]
        lons.append(lon); lats.append(lat); probs.append(pr)
grid = pd.DataFrame({"lon":lons,"lat":lats,"prob":probs})

# 7️⃣ Export artifacts
joblib.dump(rf,   "model.pkl")
grid.to_csv("grid.csv", index=False)
gdf_obs[["lon","lat"]].to_csv("obs.csv", index=False)
print("✔ model.pkl, grid.csv, obs.csv written")
