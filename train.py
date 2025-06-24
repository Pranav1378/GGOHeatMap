"""
train.py
--------
• Downloads Great-Gray-Owl observations (iNaturalist)
• Downloads Yosemite boundary & SRTM DEM
• Fetches OSM water + roads
• Trains a Random-Forest (presence vs pseudo-absence)
• Pre-computes a 500 m grid of presence probabilities
• Saves   model.pkl    grid.csv    obs.csv   into /app
"""
import io, os, zipfile, tempfile, requests, joblib
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.merge import merge
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier

########################
# 1. Owl observations  #
########################
TAXON_ID = 19890
BBOX     = "37.5,-120,38.2,-119"
url  = f"https://api.inaturalist.org/v1/observations?taxon_id={TAXON_ID}&bbox={BBOX}&quality_grade=research&per_page=200"
obs, page = [], 1
while True:
    resp = requests.get(url + f"&page={page}", timeout=30).json()
    obs.extend(resp["results"])
    if len(resp["results"]) < 200: break
    page += 1
gdf_obs = gpd.GeoDataFrame(
    geometry=[Point(o["geojson"]["coordinates"]) for o in obs],
    crs="EPSG:4326",
)
gdf_obs["lon"] = gdf_obs.geometry.x
gdf_obs["lat"] = gdf_obs.geometry.y

#############################
# 2. Yosemite park boundary #
#############################
BOUND_ZIP = "https://irma.nps.gov/DataStore/DownloadFile/693229?Reference=2301294"
zbytes    = requests.get(BOUND_ZIP, timeout=60).content
with zipfile.ZipFile(io.BytesIO(zbytes)) as z, tempfile.TemporaryDirectory() as tmp:
    z.extractall(tmp)
    shp = next(f for f in z.namelist() if f.endswith(".shp"))
    park_poly = gpd.read_file(os.path.join(tmp, shp)).unary_union

##################################
# 3. DEM mosaic + slope (SRTM 1″) #
##################################
tiles  = ["N37W120", "N38W120", "N37W119", "N38W119"]
srcs   = []
for t in tiles:
    buf = requests.get(f"https://srtm.kurviger.de/SRTM1/Region_01/{t}.hgt.zip", timeout=60).content
    with zipfile.ZipFile(io.BytesIO(buf)) as zz:
        hgt = zz.namelist()[0]
        mem = rasterio.io.MemoryFile(zz.read(hgt))
        srcs.append(mem.open(driver="SRTMHGT"))
mosaic, trans = merge(srcs)
dem = mosaic[0]

# quick-n-dirty slope
lat_c = (park_poly.bounds[1]+park_poly.bounds[3]) / 2
mpx   = 111_320 * np.cos(np.deg2rad(lat_c))
mpy   = 111_320
deg   = abs(trans.a)
dzdy  = np.gradient(dem, axis=0)/(deg*mpy)
dzdx  = np.gradient(dem, axis=1)/(deg*mpx)
slope = np.degrees(np.arctan(np.hypot(dzdy, dzdx)))

###############################
# 4. OSM water-bodies & roads #
###############################
import osmnx as ox
n,s,e,w = park_poly.bounds[3]+.02, park_poly.bounds[1]-.02, park_poly.bounds[2]+.02, park_poly.bounds[0]-.02
water = ox.geometries_from_bbox(n,s,e,w, tags={"natural":["water"], "waterway":True})
roads = ox.geometries_from_bbox(n,s,e,w, tags={"highway":True})
roads = roads[~roads.highway.isin(["path","footway","cycleway"])]

UTM = "EPSG:32611"
water_u = gpd.GeoSeries([water.unary_union], crs="EPSG:4326").to_crs(UTM).iloc[0]
road_u  = gpd.GeoSeries([roads.unary_union], crs="EPSG:4326").to_crs(UTM).iloc[0]
obs_u   = gdf_obs.to_crs(UTM)
park_u  = gpd.GeoSeries([park_poly], crs="EPSG:4326").to_crs(UTM).iloc[0]

tf_fwd  = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
tf_back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)

def feats(lat, lon):
    c = int((lon - trans.c) / trans.a)
    r = int((lat - trans.f) / trans.e)
    c = np.clip(c, 0, dem.shape[1]-1); r = np.clip(r, 0, dem.shape[0]-1)
    elev = float(dem[r, c]); slp = float(slope[r, c])
    x,y  = tf_fwd.transform(lon, lat); p = Point(x,y)
    return [elev, slp, p.distance(road_u), p.distance(water_u)]

####################################
# 5. Presence + pseudo-absence RF  #
####################################
pres = [feats(p.y, p.x) for p in gdf_obs.geometry]
dfp  = pd.DataFrame(pres, columns=["elev","slope","d_road","d_water"]); dfp["label"]=1

rng = np.random.default_rng(0)
minx,maxx,miny,maxy = park_u.bounds
bg=[]
while len(bg) < 1000:
    x,y = rng.uniform(minx,maxx), rng.uniform(miny,maxy)
    pt  = Point(x,y)
    if not park_u.contains(pt): continue
    if obs_u.distance(pt).min() < 500: continue
    lon,lat = tf_back.transform(x,y)
    bg.append(feats(lat,lon))
dfb = pd.DataFrame(bg, columns=dfp.columns); dfb["label"]=0

df  = pd.concat([dfp,dfb], ignore_index=True)
X,y = df.drop("label",axis=1).values, df.label.values
rf  = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
rf.fit(X,y)

#############################################
# 6. 500 m probability grid for heat-map    #
#############################################
xs = np.arange(minx,maxx,500); ys = np.arange(miny,maxy,500)
lons,lats,probs=[],[],[]
for x in xs:
  for y in ys:
    pt=Point(x,y)
    if not park_u.contains(pt): continue
    lon,lat = tf_back.transform(x,y)
    p = rf.predict_proba([feats(lat,lon)])[0,1]
    lons.append(lon); lats.append(lat); probs.append(p)
grid = pd.DataFrame({"lon":lons,"lat":lats,"prob":probs})

#############
# 7. Export #
#############
joblib.dump(rf, "model.pkl")
grid.to_csv("grid.csv", index=False)
gdf_obs[["lon","lat"]].to_csv("obs.csv", index=False)
print("✔  Artifacts written: model.pkl  grid.csv  obs.csv")

