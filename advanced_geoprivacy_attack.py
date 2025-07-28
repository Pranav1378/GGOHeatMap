#!/usr/bin/env python3
"""
ADVANCED GEOPRIVACY ATTACK SIMULATION
Red Team Security Assessment for iNaturalist

This tool simulates sophisticated attacks that persistent hackers might use
to break geoprivacy protections. FOR SECURITY RESEARCH ONLY.
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
import time
warnings.filterwarnings("ignore")

class AdvancedGeoprivacyAttacker:
    def __init__(self, obscured_lat, obscured_lon, accuracy_km=28.33):
        self.obscured_lat = obscured_lat
        self.obscured_lon = obscured_lon
        self.accuracy_km = accuracy_km
        
        # More aggressive assumptions about obscuring
        self.obscuring_degrees = 0.2  # Standard box
        self.box_lat_min = obscured_lat - self.obscuring_degrees/2
        self.box_lat_max = obscured_lat + self.obscuring_degrees/2
        self.box_lon_min = obscured_lon - self.obscuring_degrees/2
        self.box_lon_max = obscured_lon + self.obscuring_degrees/2
        
        print(f"🔴 ADVANCED GEOPRIVACY ATTACK SIMULATION")
        print(f"=" * 60)
        print(f"🎯 Target: Great Gray Owl sighting")
        print(f"📍 Obscured Location: {obscured_lat:.5f}, {obscured_lon:.5f}")
        print(f"⚠️  WARNING: Red team security assessment - aggressive methods")
        print()
        
    def method_1_relaxed_constraints(self):
        """
        ATTACK METHOD 1: Relax habitat constraints to be less restrictive
        """
        print(f"🔴 ATTACK METHOD 1: RELAXED HABITAT CONSTRAINTS")
        print(f"-" * 45)
        
        grid_size = 80  # Higher resolution
        lats = np.linspace(self.box_lat_min, self.box_lat_max, grid_size)
        lons = np.linspace(self.box_lon_min, self.box_lon_max, grid_size)
        
        relaxed_locations = []
        
        for lat in lats:
            for lon in lons:
                elevation = self.estimate_elevation(lat, lon)
                slope = self.estimate_slope(lat, lon)
                forest_coverage = self.estimate_forest_coverage(lat, lon)
                
                # MUCH MORE RELAXED constraints
                suitable = True
                
                # Broader elevation range (owls can be found outside "optimal" range)
                if elevation < 600 or elevation > 4200:
                    suitable = False
                
                # Allow much steeper slopes
                if slope > 50:
                    suitable = False
                
                # Much broader forest range
                if forest_coverage < 0.05:
                    suitable = False
                
                if suitable:
                    relaxed_locations.append({
                        'lat': lat,
                        'lon': lon,
                        'elevation': elevation,
                        'slope': slope,
                        'forest': forest_coverage,
                        'method': 'relaxed_constraints'
                    })
        
        self.relaxed_locations = pd.DataFrame(relaxed_locations)
        print(f"✅ Found {len(self.relaxed_locations)} locations with relaxed constraints")
        return self.relaxed_locations
    
    def method_2_ignore_all_constraints(self):
        """
        ATTACK METHOD 2: Brute force - check every location in the box
        """
        print(f"\n🔴 ATTACK METHOD 2: BRUTE FORCE - IGNORE ALL CONSTRAINTS")
        print(f"-" * 55)
        
        grid_size = 50
        lats = np.linspace(self.box_lat_min, self.box_lat_max, grid_size)
        lons = np.linspace(self.box_lon_min, self.box_lon_max, grid_size)
        
        all_locations = []
        
        for lat in lats:
            for lon in lons:
                # Accept EVERYTHING in the box - no filtering
                all_locations.append({
                    'lat': lat,
                    'lon': lon,
                    'elevation': self.estimate_elevation(lat, lon),
                    'slope': self.estimate_slope(lat, lon),
                    'forest': self.estimate_forest_coverage(lat, lon),
                    'method': 'brute_force'
                })
        
        self.all_locations = pd.DataFrame(all_locations)
        print(f"✅ Brute force found {len(self.all_locations)} total grid locations")
        return self.all_locations
    
    def method_3_probabilistic_scoring(self):
        """
        ATTACK METHOD 3: Probabilistic scoring instead of hard constraints
        """
        print(f"\n🔴 ATTACK METHOD 3: PROBABILISTIC HABITAT SCORING")
        print(f"-" * 45)
        
        if not hasattr(self, 'all_locations'):
            self.method_2_ignore_all_constraints()
        
        scored_locations = []
        
        for _, loc in self.all_locations.iterrows():
            lat, lon = loc['lat'], loc['lon']
            elevation = loc['elevation']
            slope = loc['slope']
            forest = loc['forest']
            
            # Calculate probability score (0-1) instead of binary yes/no
            prob_score = 0
            
            # Elevation probability (bell curve around optimal)
            optimal_elev = 2200
            elev_prob = np.exp(-((elevation - optimal_elev) / 800) ** 2)
            prob_score += elev_prob * 0.3
            
            # Slope probability (prefer gentle slopes but don't exclude steep)
            slope_prob = max(0.1, 1 - (slope / 45))
            prob_score += slope_prob * 0.25
            
            # Forest probability (prefer edges but don't exclude extremes)
            if 0.3 <= forest <= 0.7:
                forest_prob = 1.0
            else:
                forest_prob = 0.5  # Still possible, just less likely
            prob_score += forest_prob * 0.25
            
            # Distance to known Great Gray Owl habitat
            habitat_centers = [
                (37.8742, -119.3514),  # Tuolumne Meadows
                (37.9469, -119.7911),  # Hetch Hetchy
            ]
            
            min_habitat_dist = min([
                np.sqrt((lat - h_lat)**2 + (lon - h_lon)**2)
                for h_lat, h_lon in habitat_centers
            ])
            
            # Closer to known habitat = higher probability
            habitat_prob = max(0.1, 1 - (min_habitat_dist / 0.3))
            prob_score += habitat_prob * 0.2
            
            scored_locations.append({
                **loc.to_dict(),
                'probability_score': min(1.0, prob_score),
                'method': 'probabilistic'
            })
        
        self.scored_locations = pd.DataFrame(scored_locations)
        
        # Take top 20% of locations
        top_percentile = 0.8  # Top 20%
        threshold = self.scored_locations['probability_score'].quantile(top_percentile)
        self.high_prob_locations = self.scored_locations[
            self.scored_locations['probability_score'] >= threshold
        ].copy()
        
        print(f"✅ Probabilistic scoring identified {len(self.high_prob_locations)} high-probability locations")
        print(f"   Score threshold: {threshold:.3f}")
        print(f"   Top score: {self.scored_locations['probability_score'].max():.3f}")
        
        return self.high_prob_locations
    
    def method_4_edge_case_exploitation(self):
        """
        ATTACK METHOD 4: Exploit edge cases and unusual owl behavior
        """
        print(f"\n🔴 ATTACK METHOD 4: EDGE CASE EXPLOITATION")
        print(f"-" * 35)
        
        if not hasattr(self, 'high_prob_locations'):
            self.method_3_probabilistic_scoring()
        
        edge_locations = []
        
        for _, loc in self.high_prob_locations.iterrows():
            lat, lon = loc['lat'], loc['lon']
            elevation = loc['elevation']
            slope = loc['slope']
            
            # Look for edge cases where owls might be found
            edge_score = 0
            edge_reasons = []
            
            # Thermal updrafts (valleys and ridges)
            if elevation < 1500 or elevation > 3000:
                edge_score += 0.2
                edge_reasons.append("thermal_zone")
            
            # Steep terrain (cliff hunting perches)
            if slope > 25:
                edge_score += 0.3
                edge_reasons.append("cliff_perch")
            
            # Edge of obscuring box (might be edge of true habitat)
            dist_to_edge = min(
                abs(lat - self.box_lat_min),
                abs(lat - self.box_lat_max),
                abs(lon - self.box_lon_min),
                abs(lon - self.box_lon_max)
            )
            
            if dist_to_edge < 0.02:  # Near edge of box
                edge_score += 0.4
                edge_reasons.append("box_edge")
            
            # Transitional habitats (ecotones)
            forest = loc['forest']
            if 0.15 <= forest <= 0.35 or 0.65 <= forest <= 0.85:
                edge_score += 0.3
                edge_reasons.append("ecotone")
            
            if edge_score >= 0.3:  # Significant edge case potential
                edge_locations.append({
                    **loc.to_dict(),
                    'edge_score': edge_score,
                    'edge_reasons': edge_reasons,
                    'method': 'edge_exploitation'
                })
        
        self.edge_locations = pd.DataFrame(edge_locations)
        print(f"✅ Edge case exploitation found {len(self.edge_locations)} potential locations")
        
        return self.edge_locations
    
    def method_5_correlation_attack(self):
        """
        ATTACK METHOD 5: Multi-observation correlation attack
        """
        print(f"\n🔴 ATTACK METHOD 5: MULTI-OBSERVATION CORRELATION")
        print(f"-" * 45)
        
        # Simulate having access to multiple Great Gray Owl observations
        # This is the most dangerous attack for geoprivacy
        
        additional_observations = [
            (37.83023, -119.86828, "Target observation"),
            (37.82800, -119.87100, "Nearby observation 1"),
            (37.83400, -119.86500, "Nearby observation 2"),
            (37.83200, -119.87000, "Nearby observation 3"),
            (37.82900, -119.86900, "Nearby observation 4"),
        ]
        
        print(f"📊 Simulating correlation attack with {len(additional_observations)} observations")
        
        # Find intersection of all possible observation boxes
        overlap_locations = []
        
        if hasattr(self, 'edge_locations') and len(self.edge_locations) > 0:
            base_locations = self.edge_locations
        elif hasattr(self, 'high_prob_locations'):
            base_locations = self.high_prob_locations
        else:
            base_locations = self.all_locations
        
        for _, loc in base_locations.iterrows():
            lat, lon = loc['lat'], loc['lon']
            
            overlap_count = 0
            total_distance = 0
            
            for obs_lat, obs_lon, obs_name in additional_observations:
                # Check if this location could be within the obscuring box of each observation
                distance = np.sqrt((lat - obs_lat)**2 + (lon - obs_lon)**2)
                
                if distance <= 0.1:  # Within potential obscuring radius
                    overlap_count += 1
                
                total_distance += distance
            
            # Locations that could explain multiple observations are highly suspicious
            if overlap_count >= 3:  # Could explain 3+ observations
                correlation_score = overlap_count / len(additional_observations)
                avg_distance = total_distance / len(additional_observations)
                
                overlap_locations.append({
                    **loc.to_dict(),
                    'correlation_score': correlation_score,
                    'overlap_count': overlap_count,
                    'avg_distance': avg_distance,
                    'method': 'correlation_attack'
                })
        
        self.correlation_locations = pd.DataFrame(overlap_locations)
        print(f"✅ Correlation attack identified {len(self.correlation_locations)} high-correlation locations")
        
        if len(self.correlation_locations) > 0:
            top_correlation = self.correlation_locations.nlargest(3, 'correlation_score')
            print(f"   Top correlation scores:")
            for _, loc in top_correlation.iterrows():
                print(f"     {loc['lat']:.5f}, {loc['lon']:.5f} (score: {loc['correlation_score']:.3f})")
        
        return self.correlation_locations
    
    def estimate_elevation(self, lat, lon):
        """Estimate elevation using Yosemite topographic patterns"""
        valley_lat, valley_lon = 37.7459, -119.5936
        dist_from_valley = np.sqrt((lat - valley_lat)**2 + (lon - valley_lon)**2)
        
        if dist_from_valley < 0.02:
            elevation = 1200 + np.random.normal(0, 50)
        elif dist_from_valley < 0.1:
            elevation = 1200 + (dist_from_valley - 0.02) * 15000 + np.random.normal(0, 150)
        else:
            elevation = 2000 + (dist_from_valley - 0.1) * 20000 + np.random.normal(0, 200)
        
        return max(800, min(4200, elevation))
    
    def estimate_slope(self, lat, lon):
        """Estimate slope based on terrain"""
        valley_lat, valley_lon = 37.7459, -119.5936
        dist_from_valley = np.sqrt((lat - valley_lat)**2 + (lon - valley_lon)**2)
        
        if dist_from_valley < 0.02:
            slope = np.random.normal(2, 1)
        elif dist_from_valley < 0.05:
            slope = np.random.normal(25, 8)
        else:
            slope = np.random.normal(12, 5)
        
        return max(0, min(45, slope))
    
    def estimate_forest_coverage(self, lat, lon):
        """Estimate forest coverage"""
        elevation = self.estimate_elevation(lat, lon)
        
        if elevation < 1500:
            coverage = np.random.normal(0.7, 0.2)
        elif elevation < 2500:
            coverage = np.random.normal(0.6, 0.2)
        elif elevation < 3500:
            coverage = np.random.normal(0.4, 0.2)
        else:
            coverage = np.random.normal(0.1, 0.1)
        
        return max(0, min(1, coverage))
    
    def generate_attack_map(self):
        """Generate comprehensive attack visualization"""
        print(f"\n🗺️ GENERATING ADVANCED ATTACK MAP")
        print(f"-" * 30)
        
        m = folium.Map(
            location=[self.obscured_lat, self.obscured_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add obscured location
        folium.Marker(
            [self.obscured_lat, self.obscured_lon],
            popup="🎯 TARGET: Obscured Location",
            icon=folium.Icon(color='red', icon='bullseye', prefix='fa')
        ).add_to(m)
        
        # Add bounding box
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
            weight=3,
            opacity=0.8,
            popup="Original Obscuring Box (492.8 km²)"
        ).add_to(m)
        
        # Add attack results - final high-confidence locations
        if hasattr(self, 'correlation_locations') and len(self.correlation_locations) > 0:
            final_targets = self.correlation_locations.nlargest(5, 'correlation_score')
            for i, (_, loc) in enumerate(final_targets.iterrows()):
                folium.Marker(
                    [loc['lat'], loc['lon']],
                    popup=f"🚨 HIGH RISK TARGET #{i+1}<br>Correlation: {loc['correlation_score']:.3f}<br>{loc['lat']:.5f}, {loc['lon']:.5f}",
                    icon=folium.Icon(color='darkred', icon='exclamation-triangle', prefix='fa')
                ).add_to(m)
        
        # Add probabilistic results
        if hasattr(self, 'high_prob_locations') and len(self.high_prob_locations) > 0:
            for _, loc in self.high_prob_locations.iterrows():
                folium.CircleMarker(
                    [loc['lat'], loc['lon']],
                    radius=3,
                    popup=f"Probability: {loc.get('probability_score', 0):.3f}",
                    color='orange',
                    fillColor='orange',
                    fillOpacity=0.6
                ).add_to(m)
        
        m.save('advanced_attack_map.html')
        print(f"✅ Advanced attack map saved to 'advanced_attack_map.html'")
        
        return m
    
    def summarize_attack_results(self):
        """Summarize all attack method results"""
        print(f"\n🔴 ADVANCED GEOPRIVACY ATTACK SUMMARY")
        print(f"=" * 50)
        
        original_area = 492.8  # km²
        print(f"🎯 Target: Obscured Great Gray Owl sighting")
        print(f"📍 Original obscuring box: {original_area:.1f} km²")
        print()
        
        # Show progression of attack methods
        if hasattr(self, 'relaxed_locations'):
            print(f"🔴 Relaxed Constraints: {len(self.relaxed_locations)} locations")
        
        if hasattr(self, 'all_locations'):
            print(f"🔴 Brute Force Grid: {len(self.all_locations)} locations")
        
        if hasattr(self, 'high_prob_locations'):
            print(f"🔴 Probabilistic Scoring: {len(self.high_prob_locations)} locations")
            if len(self.high_prob_locations) > 0:
                max_score = self.high_prob_locations['probability_score'].max()
                print(f"   Maximum probability score: {max_score:.3f}")
        
        if hasattr(self, 'edge_locations'):
            print(f"🔴 Edge Case Exploitation: {len(self.edge_locations)} locations")
        
        if hasattr(self, 'correlation_locations'):
            print(f"🔴 Correlation Attack: {len(self.correlation_locations)} locations")
            
            if len(self.correlation_locations) > 0:
                top_targets = self.correlation_locations.nlargest(3, 'correlation_score')
                print(f"\n🚨 FINAL HIGH-RISK TARGETS:")
                for i, (_, loc) in enumerate(top_targets.iterrows()):
                    print(f"   #{i+1}: {loc['lat']:.5f}, {loc['lon']:.5f}")
                    print(f"        Correlation Score: {loc['correlation_score']:.3f}")
                    print(f"        Overlap Count: {loc['overlap_count']}")
                
                if len(self.correlation_locations) <= 10:
                    print(f"\n🚨 CRITICAL GEOPRIVACY VULNERABILITY!")
                    print(f"   Advanced attacks narrowed to {len(self.correlation_locations)} high-confidence targets")
                    print(f"   RECOMMEND: Immediate geoprivacy improvements")
                elif len(self.correlation_locations) <= 30:
                    print(f"\n⚠️  MODERATE GEOPRIVACY RISK")
                    print(f"   {len(self.correlation_locations)} potential targets identified")
                    print(f"   RECOMMEND: Enhanced obscuring methods")
                else:
                    print(f"\n🟡 LOW GEOPRIVACY RISK")
                    print(f"   {len(self.correlation_locations)} targets - still reasonably protected")
            else:
                print(f"\n✅ GEOPRIVACY ATTACK UNSUCCESSFUL")
                print(f"   Advanced correlation attack found no viable targets")
        else:
            print(f"\n✅ GEOPRIVACY HOLDING STRONG")
            print(f"   Even advanced attack methods could not significantly narrow location")

def main():
    print("🔴 ADVANCED GEOPRIVACY ATTACK SIMULATION")
    print("🚨 RED TEAM SECURITY ASSESSMENT - FOR RESEARCH ONLY")
    print("🦉 Target: Great Gray Owl sighting in Yosemite")
    print()
    
    # Initialize advanced attacker
    attacker = AdvancedGeoprivacyAttacker(37.83023, -119.86828, 28.33)
    
    # Execute all attack methods
    print("🔴 Executing comprehensive geoprivacy attack...")
    
    attacker.method_1_relaxed_constraints()
    attacker.method_2_ignore_all_constraints()
    attacker.method_3_probabilistic_scoring()
    attacker.method_4_edge_case_exploitation()
    attacker.method_5_correlation_attack()
    
    # Generate visualization
    attacker.generate_attack_map()
    
    # Summarize results
    attacker.summarize_attack_results()
    
    print(f"\n📋 CRITICAL SECURITY RECOMMENDATIONS:")
    print(f"1. 🚨 MONITOR CORRELATION ATTACKS - Multiple nearby observations are dangerous")
    print(f"2. 🔒 INCREASE OBSCURING BOX SIZE for rare species (0.3° or larger)")
    print(f"3. ⏰ ADD TEMPORAL OBSCURING - randomize observation dates/times")
    print(f"4. 🎲 DYNAMIC OBSCURING - vary box size based on habitat specificity")
    print(f"5. 📊 PROBABILISTIC OBSCURING - add noise to reduce correlation attacks")
    print(f"6. 🚫 METADATA LIMITATION - restrict access to observation details")

if __name__ == "__main__":
    main() 