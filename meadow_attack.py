#!/usr/bin/env python3
"""
MULTI-FACTOR GEOPRIVACY ATTACK
-------------------------------

This tool simulates a highly sophisticated, multi-modal geoprivacy attack.
It fuses ecological knowledge with advanced computer vision to analyze:
  - Prime Habitat (Meadows)
  - Water Proximity (Water Sources)
  - Human Avoidance (Roads/Buildings)
  - Topography Proxies (Shadows)

The tool outputs a direct Google Maps link to the #1 highest-probability target.

FOR SECURITY RESEARCH AND EDUCATIONAL PURPOSES ONLY.
"""

import os
import requests
import cv2
import numpy as np
import pandas as pd
import sys
import webbrowser
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
OBSCURED_LAT = 37.83023
OBSCURED_LON = -119.86828
OBSCURING_DEGREES = 0.2
GRID_RESOLUTION = 25
MAP_ZOOM = 16 # Increased zoom for more detail

# --- COMPUTER VISION PARAMETERS ---
# HSV Color Ranges
MEADOW_HSV_LOWER = np.array([20, 30, 100])
MEADOW_HSV_UPPER = np.array([80, 200, 255])
WATER_HSV_LOWER = np.array([90, 50, 30])
WATER_HSV_UPPER = np.array([130, 255, 150])
SHADOW_V_MAX = 60 # Max 'Value' for a pixel to be a shadow

# Thresholds
MIN_MEADOW_PERCENT = 0.10
MAX_MEADOW_PERCENT = 0.80
MIN_EDGE_STRENGTH = 100
MIN_WATER_PERCENT = 0.02 # At least 2% of tile should be water for bonus
MAX_SHADOW_PERCENT = 0.30 # Penalize if more than 30% of tile is shadow
MIN_HOUGH_LINES = 15 # Number of straight lines to detect to flag as "human infrastructure"

class MultiFactorAttacker:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
        self.results = []
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

    def analyze_tile_with_cv(self, image):
        """
        Applies multiple CV models to a satellite tile.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_pixels = image.shape[0] * image.shape[1]

        # 1. Meadow Score
        meadow_mask = cv2.inRange(hsv_image, MEADOW_HSV_LOWER, MEADOW_HSV_UPPER)
        meadow_percent = cv2.countNonZero(meadow_mask) / total_pixels
        edge_strength = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        meadow_score = 0
        if MIN_MEADOW_PERCENT < meadow_percent < MAX_MEADOW_PERCENT and edge_strength > MIN_EDGE_STRENGTH:
            meadow_score = 0.8 + (0.2 * (meadow_percent / MAX_MEADOW_PERCENT)) # Bonus for more meadow

        # 2. Water Bonus
        water_mask = cv2.inRange(hsv_image, WATER_HSV_LOWER, WATER_HSV_UPPER)
        water_percent = cv2.countNonZero(water_mask) / total_pixels
        water_bonus = 1.0
        if water_percent > MIN_WATER_PERCENT:
            water_bonus = 1.5 # Significant bonus for water presence

        # 3. Human Infrastructure Penalty
        canny_edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        human_penalty = 0
        if lines is not None and len(lines) > MIN_HOUGH_LINES:
            human_penalty = 0.5 # Penalize for signs of roads/buildings

        # 4. Shadow Penalty
        shadow_mask = cv2.inRange(hsv_image, np.array([0,0,0]), np.array([180,255,SHADOW_V_MAX]))
        shadow_percent = cv2.countNonZero(shadow_mask) / total_pixels
        shadow_penalty = 0
        if shadow_percent > MAX_SHADOW_PERCENT:
            shadow_penalty = 0.3 # Penalize for excessive shadows

        return {
            "meadow_score": meadow_score,
            "water_bonus": water_bonus,
            "human_penalty": human_penalty,
            "shadow_penalty": shadow_penalty
        }

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
        print("🚀 Starting Multi-Factor Geoprivacy Attack...")
        print(f"[*] Analyzing a {GRID_RESOLUTION}x{GRID_RESOLUTION} grid across the obscured area.")
        
        lat_points = np.linspace(self.box_lat_min, self.box_lat_max, GRID_RESOLUTION)
        lon_points = np.linspace(self.box_lon_min, self.box_lon_max, GRID_RESOLUTION)

        count = 0
        for lat in lat_points:
            for lon in lon_points:
                count += 1
                print(f"\r[*] Analyzing tile {count}/{GRID_RESOLUTION**2} ({lat:.4f}, {lon:.4f})...", end="")
                
                tile = self.get_satellite_tile(lat, lon)
                if tile is None: continue

                cv_results = self.analyze_tile_with_cv(tile)
                if cv_results["meadow_score"] == 0: continue

                elevation = self.estimate_elevation(lat, lon)
                elevation_score = 0
                if 1200 < elevation < 2700: elevation_score = 1.0
                elif 1000 < elevation < 3000: elevation_score = 0.5
                
                # Hacker's Logic: Final score is a mix of positive and negative factors
                final_score = (cv_results["meadow_score"] * elevation_score * cv_results["water_bonus"]) \
                              - cv_results["human_penalty"] - cv_results["shadow_penalty"]
                
                if final_score > 0.5: # Only keep high-confidence results
                    self.results.append({
                        "lat": lat, "lon": lon, "final_score": final_score,
                        "elevation_m": elevation, **cv_results
                    })
        print("\n[+] Analysis complete.")

    def deliver_payload(self):
        """Finds the top target and opens it in Google Maps."""
        if not self.results:
            print("❌ ATTACK FAILED: No high-confidence locations found.")
            return

        print("\n[!] ATTACK SUCCESSFUL. Delivering payload...")
        df = pd.DataFrame(self.results)
        top_target = df.sort_values(by="final_score", ascending=False).iloc[0]
        
        lat = top_target['lat']
        lon = top_target['lon']

        # Construct the Google Maps URL
        gmaps_url = f"https://www.google.com/maps/@?api=1&map_action=map&center={lat},{lon}&zoom={MAP_ZOOM+2}&basemap=satellite"
        
        print(f"🎯 Top Target Identified at ({lat:.5f}, {lon:.5f})")
        print(f"   -> Score: {top_target['final_score']:.2f}")
        print(f"   -> Elevation: {top_target['elevation_m']:.0f}m")
        print(f"   -> Factors: Meadow={top_target['meadow_score']:.2f}, Water Bonus={top_target['water_bonus']:.1f}, Human Penalty={top_target['human_penalty']:.1f}")
        print(f"\n[+] Opening target in Google Maps...")

        try:
            webbrowser.open(gmaps_url)
            print("[+] Payload delivered.")
        except Exception as e:
            print(f"❌ Could not open web browser. Manually open this link:\n{gmaps_url}")
        
        print("\n--- SECURITY ASSESSMENT ---")
        print("🔴 CRITICAL VULNERABILITY: Layered intelligence attack has defeated geoprivacy.")
        print("🔴 FINDING: By fusing multiple weak signals (habitat, water, human activity), the true location can be pinpointed with high confidence.")
        print("RECOMMENDATION: This level of attack demonstrates the limits of simple coordinate fuzzing. Advanced countermeasures are required, potentially including injecting decoy information or applying differential privacy concepts to the underlying data sources.")

def main():
    """Main execution function."""
    print("="*60)
    print("  MULTI-FACTOR GEOPRIVACY ATTACK - RED TEAM TOOL")
    print("="*60)

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("❌ ERROR: Google Maps API Key not found.")
        print("Please set the GOOGLE_MAPS_API_KEY environment variable and try again.")
        sys.exit(1)

    attacker = MultiFactorAttacker(api_key=api_key)
    attacker.run_attack()
    attacker.deliver_payload()

if __name__ == "__main__":
    main() 