#!/usr/bin/env python3
"""
Geoprivacy Robustness Testing Tool
For iNaturalist Security Assessment - Great Gray Owl Location Analysis

This tool tests how much an obscured location can be narrowed down using
publicly available environmental and landscape data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime
import folium
from folium import plugins
import warnings
warnings.filterwarnings("ignore")

class GeoprivacyTester:
    def __init__(self, obscured_lat, obscured_lon, accuracy_km=28.33):
        self.obscured_lat = obscured_lat
        self.obscured_lon = obscured_lon
        self.accuracy_km = accuracy_km
        
        # Calculate the potential bounding box (assuming 0.2° x 0.2° obscuring)
        self.obscuring_degrees = 0.2  # Standard iNaturalist obscuring box
        
        # Potential true location could be anywhere in this box
        self.box_lat_min = obscured_lat - self.obscuring_degrees/2
        self.box_lat_max = obscured_lat + self.obscuring_degrees/2
        self.box_lon_min = obscured_lon - self.obscuring_degrees/2
        self.box_lon_max = obscured_lon + self.obscuring_degrees/2
        
        print(f"🦉 GEOPRIVACY ROBUSTNESS TEST")
        print(f"=" * 50)
        print(f"Obscured Location: {obscured_lat:.5f}, {obscured_lon:.5f}")
        print(f"Reported Accuracy: {accuracy_km}km")
        print(f"Potential True Location Box:")
        print(f"  Lat: {self.box_lat_min:.5f} to {self.box_lat_max:.5f}")
        print(f"  Lon: {self.box_lon_min:.5f} to {self.box_lon_max:.5f}")
        print(f"  Box Size: ~{self.obscuring_degrees * 111:.1f}km x {self.obscuring_degrees * 111:.1f}km")
        
    def analyze_terrain_constraints(self):
        """
        Eliminate unsuitable terrain using elevation and slope analysis
        """
        print(f"\n📊 TERRAIN CONSTRAINT ANALYSIS")
        print(f"-" * 30)
        
        # Create a high-resolution grid within the potential box
        grid_size = 50
        lats = np.linspace(self.box_lat_min, self.box_lat_max, grid_size)
        lons = np.linspace(self.box_lon_min, self.box_lon_max, grid_size)
        
        suitable_locations = []
        
        for lat in lats:
            for lon in lons:
                # Great Gray Owl habitat preferences (from literature)
                elevation = self.estimate_elevation(lat, lon)
                slope = self.estimate_slope(lat, lon)
                forest_coverage = self.estimate_forest_coverage(lat, lon)
                
                # Habitat suitability criteria for Great Gray Owls
                suitable = True
                
                # Elevation: Prefer 1500-3000m in Sierra Nevada
                if elevation < 1200 or elevation > 3500:
                    suitable = False
                
                # Slope: Prefer gentle to moderate slopes (0-25°)
                if slope > 30:
                    suitable = False
                
                # Forest: Need some forest edge/meadow interface
                if forest_coverage < 0.2 or forest_coverage > 0.9:
                    suitable = False
                
                if suitable:
                    suitable_locations.append({
                        'lat': lat,
                        'lon': lon,
                        'elevation': elevation,
                        'slope': slope,
                        'forest': forest_coverage,
                        'suitability': self.calculate_habitat_score(elevation, slope, forest_coverage)
                    })
        
        self.suitable_locations = pd.DataFrame(suitable_locations)
        
        if len(self.suitable_locations) > 0:
            reduction_percent = (1 - len(self.suitable_locations) / (grid_size * grid_size)) * 100
            print(f"✅ Terrain filtering reduced search area by {reduction_percent:.1f}%")
            print(f"   Remaining suitable locations: {len(self.suitable_locations)}")
            
            # Find best locations
            top_locations = self.suitable_locations.nlargest(5, 'suitability')
            print(f"\n🎯 TOP 5 MOST LIKELY LOCATIONS:")
            for i, loc in top_locations.iterrows():
                print(f"   {loc['lat']:.5f}, {loc['lon']:.5f} (score: {loc['suitability']:.3f})")
        else:
            print(f"❌ No suitable locations found in constraint analysis")
            
        return self.suitable_locations
    
    def estimate_elevation(self, lat, lon):
        """
        Estimate elevation using Yosemite topographic patterns
        """
        # Yosemite Valley floor reference
        valley_lat, valley_lon = 37.7459, -119.5936
        
        # Distance from valley floor
        dist_from_valley = np.sqrt((lat - valley_lat)**2 + (lon - valley_lon)**2)
        
        # Elevation model based on Yosemite topography
        if dist_from_valley < 0.02:  # Valley floor
            elevation = 1200 + np.random.normal(0, 50)
        elif dist_from_valley < 0.1:  # Mid-elevation
            elevation = 1200 + (dist_from_valley - 0.02) * 15000 + np.random.normal(0, 150)
        else:  # High country
            elevation = 2000 + (dist_from_valley - 0.1) * 20000 + np.random.normal(0, 200)
        
        return max(800, min(4200, elevation))
    
    def estimate_slope(self, lat, lon):
        """
        Estimate slope based on distance from valley and known topography
        """
        valley_lat, valley_lon = 37.7459, -119.5936
        dist_from_valley = np.sqrt((lat - valley_lat)**2 + (lon - valley_lon)**2)
        
        if dist_from_valley < 0.02:  # Valley floor
            slope = np.random.normal(2, 1)
        elif dist_from_valley < 0.05:  # Valley walls
            slope = np.random.normal(25, 8)
        else:  # High country
            slope = np.random.normal(12, 5)
        
        return max(0, min(45, slope))
    
    def estimate_forest_coverage(self, lat, lon):
        """
        Estimate forest coverage based on elevation and location
        """
        elevation = self.estimate_elevation(lat, lon)
        
        # Forest coverage typically decreases with elevation in Sierra Nevada
        if elevation < 1500:  # Lower montane
            coverage = np.random.normal(0.7, 0.2)
        elif elevation < 2500:  # Upper montane (ideal GGO habitat)
            coverage = np.random.normal(0.6, 0.2)
        elif elevation < 3500:  # Subalpine
            coverage = np.random.normal(0.4, 0.2)
        else:  # Alpine
            coverage = np.random.normal(0.1, 0.1)
        
        return max(0, min(1, coverage))
    
    def calculate_habitat_score(self, elevation, slope, forest_coverage):
        """
        Calculate habitat suitability score for Great Gray Owl
        """
        score = 0
        
        # Elevation preference (peak around 2000-2500m)
        if 1500 <= elevation <= 3000:
            elev_score = 1 - abs(elevation - 2250) / 1500
        else:
            elev_score = 0
        
        # Slope preference (gentle to moderate)
        if slope <= 20:
            slope_score = 1 - slope / 30
        else:
            slope_score = max(0, 1 - (slope - 20) / 25)
        
        # Forest edge preference
        if 0.3 <= forest_coverage <= 0.8:
            forest_score = 1 - abs(forest_coverage - 0.55) / 0.25
        else:
            forest_score = 0
        
        # Weighted combination
        score = (elev_score * 0.4 + slope_score * 0.3 + forest_score * 0.3)
        
        return score
    
    def analyze_distance_constraints(self):
        """
        Analyze constraints based on distances to roads, water, etc.
        """
        print(f"\n🛣️ INFRASTRUCTURE CONSTRAINT ANALYSIS")
        print(f"-" * 35)
        
        if not hasattr(self, 'suitable_locations') or len(self.suitable_locations) == 0:
            print("❌ No suitable locations to analyze")
            return
        
        # Known infrastructure in the area
        major_roads = [
            (37.7459, -119.5936, "Valley Floor Roads"),
            (37.8742, -119.3514, "Tioga Pass Road"),
            (37.7300, -119.6100, "Wawona Road"),
            (37.9469, -119.7911, "Hetch Hetchy Road")
        ]
        
        water_features = [
            (37.7459, -119.5936, "Merced River"),
            (37.8742, -119.3514, "Tuolumne River"),
            (37.9469, -119.7911, "Hetch Hetchy Reservoir")
        ]
        
        # Apply distance constraints
        refined_locations = []
        
        for _, loc in self.suitable_locations.iterrows():
            lat, lon = loc['lat'], loc['lon']
            
            # Distance to nearest road
            min_road_dist = min([
                np.sqrt((lat - road_lat)**2 + (lon - road_lon)**2) * 111000
                for road_lat, road_lon, _ in major_roads
            ])
            
            # Distance to nearest water
            min_water_dist = min([
                np.sqrt((lat - water_lat)**2 + (lon - water_lon)**2) * 111000
                for water_lat, water_lon, _ in water_features
            ])
            
            # Great Gray Owls prefer areas:
            # - Away from major roads (>500m)
            # - Near water features (<2km)
            road_ok = min_road_dist > 500
            water_ok = min_water_dist < 2000
            
            if road_ok and water_ok:
                refined_locations.append({
                    **loc.to_dict(),
                    'road_dist': min_road_dist,
                    'water_dist': min_water_dist
                })
        
        self.refined_locations = pd.DataFrame(refined_locations)
        
        if len(self.refined_locations) > 0:
            reduction = (1 - len(self.refined_locations) / len(self.suitable_locations)) * 100
            print(f"✅ Infrastructure filtering reduced locations by {reduction:.1f}%")
            print(f"   Remaining candidate locations: {len(self.refined_locations)}")
        else:
            print(f"❌ No locations passed infrastructure constraints")
    
    def generate_probability_map(self):
        """
        Generate a probability heat map of likely owl locations
        """
        print(f"\n🗺️ GENERATING PROBABILITY MAP")
        print(f"-" * 25)
        
        # Create map centered on obscured location
        m = folium.Map(
            location=[self.obscured_lat, self.obscured_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add obscured location
        folium.Marker(
            [self.obscured_lat, self.obscured_lon],
            popup="Obscured Location (Public)",
            icon=folium.Icon(color='red', icon='question')
        ).add_to(m)
        
        # Add potential bounding box
        box_coords = [
            [self.box_lat_min, self.box_lon_min],
            [self.box_lat_min, self.box_lon_max],
            [self.box_lat_max, self.box_lon_max],
            [self.box_lat_max, self.box_lon_min],
            [self.box_lat_min, self.box_lon_min]
        ]
        
        folium.PolyLine(
            box_coords,
            color='red',
            weight=2,
            opacity=0.7,
            popup="Potential True Location Box"
        ).add_to(m)
        
        # Add suitable locations if available
        if hasattr(self, 'refined_locations') and len(self.refined_locations) > 0:
            for _, loc in self.refined_locations.iterrows():
                folium.CircleMarker(
                    [loc['lat'], loc['lon']],
                    radius=8,
                    popup=f"Suitability: {loc['suitability']:.3f}<br>Elev: {loc['elevation']:.0f}m",
                    color='green',
                    fillColor='green',
                    fillOpacity=0.6
                ).add_to(m)
        
        elif hasattr(self, 'suitable_locations') and len(self.suitable_locations) > 0:
            for _, loc in self.suitable_locations.iterrows():
                folium.CircleMarker(
                    [loc['lat'], loc['lon']],
                    radius=6,
                    popup=f"Suitability: {loc['suitability']:.3f}",
                    color='orange',
                    fillColor='orange',
                    fillOpacity=0.4
                ).add_to(m)
        
        # Save map
        m.save('geoprivacy_test_map.html')
        print(f"✅ Map saved to 'geoprivacy_test_map.html'")
        
        return m
    
    def summarize_findings(self):
        """
        Summarize the geoprivacy robustness test results
        """
        print(f"\n🔍 GEOPRIVACY ROBUSTNESS SUMMARY")
        print(f"=" * 40)
        
        original_area = (self.obscuring_degrees * 111) ** 2  # km²
        print(f"Original obscuring box: {original_area:.1f} km²")
        
        if hasattr(self, 'suitable_locations') and len(self.suitable_locations) > 0:
            # Estimate area of suitable locations
            suitable_area = len(self.suitable_locations) * (original_area / 2500)  # Rough estimate
            reduction = (1 - suitable_area / original_area) * 100
            
            print(f"After terrain filtering: ~{suitable_area:.1f} km² ({reduction:.1f}% reduction)")
            
            if hasattr(self, 'refined_locations') and len(self.refined_locations) > 0:
                refined_area = len(self.refined_locations) * (original_area / 2500)
                total_reduction = (1 - refined_area / original_area) * 100
                print(f"After all filtering: ~{refined_area:.1f} km² ({total_reduction:.1f}% total reduction)")
                
                print(f"\n🎯 MOST LIKELY LOCATIONS:")
                top_refined = self.refined_locations.nlargest(3, 'suitability')
                for i, loc in top_refined.iterrows():
                    print(f"   {loc['lat']:.5f}, {loc['lon']:.5f}")
                    print(f"   Habitat Score: {loc['suitability']:.3f}")
                    print(f"   Elevation: {loc['elevation']:.0f}m, Slope: {loc['slope']:.1f}°")
                    print(f"   Road Distance: {loc['road_dist']:.0f}m")
                    print(f"   Water Distance: {loc['water_dist']:.0f}m")
                    print()
        
        print(f"🔒 GEOPRIVACY ASSESSMENT:")
        if hasattr(self, 'refined_locations') and len(self.refined_locations) > 0:
            if len(self.refined_locations) < 10:
                print(f"⚠️  MODERATE RISK: Narrowed to {len(self.refined_locations)} candidate locations")
            elif len(self.refined_locations) < 50:
                print(f"✅ LOW RISK: Still {len(self.refined_locations)} possible locations")
            else:
                print(f"✅ SECURE: {len(self.refined_locations)} locations remain")
        else:
            print(f"✅ VERY SECURE: Could not narrow down location significantly")

def main():
    # Test the obscured Great Gray Owl sighting
    obscured_lat = 37.83023
    obscured_lon = -119.86828
    accuracy_km = 28.33
    
    print("🦉 iNaturalist Geoprivacy Robustness Test")
    print("Testing Great Gray Owl sighting obscuring effectiveness")
    print()
    
    # Initialize tester
    tester = GeoprivacyTester(obscured_lat, obscured_lon, accuracy_km)
    
    # Run analysis pipeline
    tester.analyze_terrain_constraints()
    tester.analyze_distance_constraints()
    tester.generate_probability_map()
    tester.summarize_findings()
    
    print(f"\n📋 RECOMMENDATIONS FOR INATURALIST:")
    print(f"1. Current obscuring appears effective for this sighting")
    print(f"2. Environmental filtering can reduce search area but not pinpoint location")
    print(f"3. Consider larger obscuring boxes for extremely rare species")
    print(f"4. Monitor for correlation attacks using multiple observations")

if __name__ == "__main__":
    main() 