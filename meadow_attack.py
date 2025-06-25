#!/usr/bin/env python3
"""
MEADOW HUNTER - ADVANCED GEOPRIVACY ATTACK
-------------------------------------------

This tool simulates an advanced, multi-modal geoprivacy attack by
fusing geospatial data with computer vision analysis of satellite imagery
to identify specific habitat features (high-elevation meadows).

FOR SECURITY RESEARCH AND EDUCATIONAL PURPOSES ONLY.
"""

import os
import requests
import cv2
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import warnings
import sys
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
OBSCURED_LAT = 37.83023
OBSCURED_LON = -119.86828
OBSCURING_DEGREES = 0.2  # Approx. 22.2 km box
GRID_RESOLUTION = 25     # How many points to check across the box (25x25 = 625 tiles)
MAP_ZOOM = 15            # Google Maps zoom level for satellite tiles

# --- COMPUTER VISION PARAMETERS for MEADOW DETECTION ---
# Color range for meadows in HSV color space (Hue, Saturation, Value)
# These values target light greens and tans, common in Sierra meadows.
MEADOW_HSV_LOWER = np.array([20, 30, 100])
MEADOW_HSV_UPPER = np.array([80, 200, 255])

# Thresholds for what constitutes a "good" meadow tile
MIN_MEADOW_PERCENT = 0.10  # At least 10% of the image should be meadow-like
MAX_MEADOW_PERCENT = 0.80  # But not more than 80% (to find edges, not huge open fields)
MIN_EDGE_STRENGTH = 100    # How "textured" the image is (high value means forest edges)

class MeadowHunter:
    """
    Orchestrates the meadow hunting attack.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
        self.results = []

        # Define the bounding box for the attack
        self.box_lat_min = OBSCURED_LAT - OBSCURING_DEGREES / 2
        self.box_lat_max = OBSCURED_LAT + OBSCURING_DEGREES / 2
        self.box_lon_min = OBSCURED_LON - OBSCURING_DEGREES / 2
        self.box_lon_max = OBSCURED_LON + OBSCURING_DEGREES / 2

    def get_satellite_tile(self, lat, lon):
        """Downloads a satellite image tile for a given coordinate."""
        params = {
            "center": f"{lat},{lon}",
            "zoom": MAP_ZOOM,
            "size": "400x400",
            "maptype": "satellite",
            "key": self.api_key,
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            image_data = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                print(f"⚠️  Warning: Could not decode image for {lat},{lon}")
                return None
            return image
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Failed to download satellite image: {e}")
            # Check for common API key issues
            if "key" in str(e).lower() or "denied" in str(e).lower():
                print("    -> This might be an API key issue. Is it valid and enabled?")
            return None

    def analyze_tile_for_meadows(self, image):
        """
        Uses computer vision to detect meadows and forest edges in a satellite tile.
        Returns a 'meadow_score' from 0 to 1.
        """
        # 1. Convert to HSV color space for better color isolation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Create a mask to find pixels within the meadow color range
        meadow_mask = cv2.inRange(hsv_image, MEADOW_HSV_LOWER, MEADOW_HSV_UPPER)

        # 3. Calculate meadow percentage
        meadow_pixel_count = cv2.countNonZero(meadow_mask)
        total_pixels = image.shape[0] * image.shape[1]
        meadow_percent = meadow_pixel_count / total_pixels

        # 4. Calculate edge strength (texture) using Laplacian variance
        # High variance indicates complex textures like forest canopies and edges.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_strength = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        # 5. Score the tile based on our criteria
        meadow_score = 0
        if (MIN_MEADOW_PERCENT < meadow_percent < MAX_MEADOW_PERCENT and 
            edge_strength > MIN_EDGE_STRENGTH):
            # This is a prime candidate: a good amount of meadow with strong edges
            meadow_score = 0.8  # High base score
            # Bonus for being in the "sweet spot" of meadow percentage
            if 0.2 < meadow_percent < 0.6:
                meadow_score = 1.0
        
        return meadow_score, meadow_percent, edge_strength

    def estimate_elevation(self, lat, lon):
        """
        Dummy function to estimate elevation. A real attacker would use a
        service like Open Topo Data API, but this simulates the pattern.
        """
        # Yosemite elevation rises from west to east
        base_elevation = 1000 + (lon - self.box_lon_min) * 15000
        # Add some noise
        elevation = base_elevation + np.random.normal(0, 200)
        return max(500, min(4000, elevation))

    def run_attack(self):
        """Executes the full grid analysis."""
        print("🛰️  Starting Meadow Hunter Attack...")
        print(f"[*] Analyzing a {GRID_RESOLUTION}x{GRID_RESOLUTION} grid across the obscured area.")
        
        lat_points = np.linspace(self.box_lat_min, self.box_lat_max, GRID_RESOLUTION)
        lon_points = np.linspace(self.box_lon_min, self.box_lon_max, GRID_RESOLUTION)

        count = 0
        for lat in lat_points:
            for lon in lon_points:
                count += 1
                print(f"\r[*] Analyzing tile {count}/{GRID_RESOLUTION**2} at ({lat:.4f}, {lon:.4f})...", end="")
                
                # 1. Get satellite image
                tile = self.get_satellite_tile(lat, lon)
                if tile is None:
                    continue

                # 2. Analyze with CV
                meadow_score, meadow_percent, edge_strength = self.analyze_tile_for_meadows(tile)
                
                # 3. Get elevation
                elevation = self.estimate_elevation(lat, lon)
                
                # 4. Calculate elevation suitability (prefers 1200m-2700m)
                elevation_score = 0
                if 1200 < elevation < 2700:
                    elevation_score = 1.0
                elif 1000 < elevation < 3000:
                    elevation_score = 0.5
                
                # 5. Final Score = Meadow Score * Elevation Score
                final_score = meadow_score * elevation_score
                
                if final_score > 0:
                    self.results.append({
                        "lat": lat,
                        "lon": lon,
                        "meadow_score": meadow_score,
                        "elevation_score": elevation_score,
                        "final_score": final_score,
                        "elevation_m": elevation,
                        "meadow_percent": meadow_percent,
                        "edge_strength": edge_strength,
                    })
        print("\n[+] Analysis complete.")
    
    def generate_report_and_map(self):
        """Generates a final map and summary report."""
        if not self.results:
            print("❌ No potential meadow locations found. The area may be unsuitable or the CV parameters need tuning.")
            return

        print("\n🗺️  Generating attack map and report...")
        df = pd.DataFrame(self.results)
        
        # Create base map with satellite imagery as the default
        m = folium.Map(
            location=[OBSCURED_LAT, OBSCURED_LON], 
            zoom_start=11, 
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community"
        )

        # Add other tile layers for context
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)

        # Add heatmap layer
        heat_data = [[row['lat'], row['lon'], row['final_score']] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=12).add_to(m)
        
        # Add top 5 candidates as markers
        top_candidates = df.sort_values(by="final_score", ascending=False).head(5)
        for i, row in top_candidates.iterrows():
            popup_html = f"""
            <b>🎯 High-Priority Target #{i+1}</b><br>
            ---------------------------<br>
            <b>Coords:</b> {row['lat']:.4f}, {row['lon']:.4f}<br>
            <b>Final Score:</b> {row['final_score']:.2f}<br>
            <b>Elevation:</b> {row['elevation_m']:.0f} m<br>
            <b>Meadow %:</b> {row['meadow_percent']:.2%}<br>
            <b>Edge Strength:</b> {row['edge_strength']:.0f}
            """
            folium.Marker(
                [row['lat'], row['lon']],
                popup=popup_html,
                icon=folium.Icon(color="red", icon="star")
            ).add_to(m)

        # Add a layer control to switch between map types
        folium.LayerControl().add_to(m)

        map_path = "meadow_attack_map.html"
        m.save(map_path)
        print(f"[+] Attack map saved to: {map_path}")

        # Print summary report
        print("\n--- MEADOW HUNTER ATTACK SUMMARY ---")
        print(f"Total Tiles Analyzed: {GRID_RESOLUTION**2}")
        print(f"Potential Meadow Locations Found: {len(df)}")
        print("\n--- TOP 5 CANDIDATE LOCATIONS ---")
        print(top_candidates[['lat', 'lon', 'final_score', 'elevation_m']].to_string(index=False))
        print("\n--- SECURITY ASSESSMENT ---")
        print("🔴 VULNERABILITY: Multi-modal data fusion (CV + Geospatial) can effectively defeat simple obscuring.")
        print("🔴 FINDING: By targeting specific habitat features (meadows), we reduced the search space from ~490 km² to a few high-probability zones.")
        print("RECOMMENDATION: True geoprivacy requires not just coordinate fuzzing, but also countermeasures against feature-based attacks, such as adding noise or misdirection to habitat data itself.")

def main():
    """Main execution function."""
    print("="*50)
    print("  MEADOW HUNTER - GEOPRIVACY RED TEAM TOOL")
    print("="*50)

    # A real hacker would not be this nice.
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("❌ ERROR: Google Maps API Key not found.")
        print("Please set the GOOGLE_MAPS_API_KEY environment variable.")
        print("\nTo get a key:")
        print("1. Go to Google Cloud Console (console.cloud.google.com)")
        print("2. Create a new project.")
        print("3. Enable 'Maps Static API'.")
        print("4. Create an API key under 'Credentials'.")
        print("5. IMPORTANT: Secure your API key to prevent unauthorized use!")
        sys.exit(1)

    hunter = MeadowHunter(api_key=api_key)
    hunter.run_attack()
    hunter.generate_report_and_map()

if __name__ == "__main__":
    main() 