#date: 2025-09-26T17:02:51Z
#url: https://api.github.com/gists/fd408e3d3927b80e1597a5b0936cefc1
#owner: https://api.github.com/users/icepuente

#!/usr/bin/env python3
"""
Convert GPX route to track format with distance calculation for Stryd compatibility.
"""

import xml.etree.ElementTree as ET
import math
import requests
import time
from datetime import datetime, timedelta

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in meters."""
    R = 6371000  # Earth's radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def get_elevation(lat, lon, use_api=True):
    """Get elevation for a given coordinate using Open Elevation API or approximation."""
    if not use_api:
        # Fallback: approximate elevation for Phoenix area (300-400m)
        return 350.0

    try:
        # Using Open Elevation API (free, no API key required)
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data['results']:
                elevation = data['results'][0]['elevation']
                return float(elevation) if elevation is not None else 350.0

        # Fallback if API fails
        return 350.0

    except Exception as e:
        print(f"Warning: Could not get elevation for {lat},{lon}: {e}")
        return 350.0

def get_elevations_batch(coordinates, use_api=True, batch_size=50):
    """Get elevations for multiple coordinates, with batching for API efficiency."""
    elevations = []

    if not use_api:
        # Return approximate elevations for Phoenix area
        return [350.0] * len(coordinates)

    try:
        print(f"Fetching elevation data for {len(coordinates)} points in batches of {batch_size}...")

        # Split coordinates into smaller batches to avoid URI too long error
        for i in range(0, len(coordinates), batch_size):
            batch = coordinates[i:i + batch_size]
            locations = "|".join([f"{lat},{lon}" for lat, lon in batch])
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

            print(f"  Batch {i//batch_size + 1}/{(len(coordinates) + batch_size - 1)//batch_size}...")
            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                for result in data['results']:
                    elevation = result['elevation']
                    elevations.append(float(elevation) if elevation is not None else 350.0)
            else:
                print(f"  Batch failed with status {response.status_code}, using fallback for this batch")
                elevations.extend([350.0] * len(batch))

            # Small delay between requests to be nice to the API
            time.sleep(0.1)

    except Exception as e:
        print(f"Warning: Batch elevation request failed: {e}")
        print("Using fallback elevation values")
        elevations = [350.0] * len(coordinates)

    return elevations

def convert_route_to_track(input_file, output_file, use_elevation_api=True):
    """Convert GPX route to track format with distance metadata."""

    # Parse the input GPX file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find the route points
    route_points = []
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    for rtept in root.findall('.//gpx:rtept', ns):
        lat = float(rtept.get('lat'))
        lon = float(rtept.get('lon'))
        route_points.append((lat, lon))

    if not route_points:
        raise ValueError("No route points found in GPX file")

    print(f"Found {len(route_points)} route points")

    # Get elevation data for all points
    print("Fetching elevation data...")
    elevations = get_elevations_batch(route_points, use_elevation_api)
    print(f"Retrieved elevation data: min={min(elevations):.1f}m, max={max(elevations):.1f}m")

    # Calculate total distance
    total_distance = 0
    for i in range(1, len(route_points)):
        lat1, lon1 = route_points[i-1]
        lat2, lon2 = route_points[i]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        total_distance += distance

    print(f"Calculated total distance: {total_distance/1000:.2f} km ({total_distance/1609.34:.2f} miles)")

    # Check if distance is within Stryd's acceptable range (1.5k - 50k)
    if total_distance < 1500:
        print("WARNING: Distance is less than 1.5km - Stryd may reject this file")
    elif total_distance > 50000:
        print("WARNING: Distance is greater than 50km - Stryd may reject this file")

    # Create new GPX with track format
    gpx_root = ET.Element('gpx')
    gpx_root.set('version', '1.1')
    gpx_root.set('creator', 'GPX Distance Fixer for Stryd')
    gpx_root.set('xmlns', 'http://www.topografix.com/GPX/1/1')
    gpx_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    gpx_root.set('xsi:schemaLocation', 'http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd')

    # Add metadata
    metadata = ET.SubElement(gpx_root, 'metadata')
    name_elem = ET.SubElement(metadata, 'name')
    name_elem.text = "Marathon Course (Fixed for Stryd)"
    desc_elem = ET.SubElement(metadata, 'desc')
    desc_elem.text = f"Total distance: {total_distance/1000:.2f} km"

    # Create track
    trk = ET.SubElement(gpx_root, 'trk')
    trk_name = ET.SubElement(trk, 'name')
    trk_name.text = "Marathon Course"

    # Create track segment
    trkseg = ET.SubElement(trk, 'trkseg')

    # Add track points with timestamps and elevation
    start_time = datetime.now()
    cumulative_distance = 0

    for i, (lat, lon) in enumerate(route_points):
        trkpt = ET.SubElement(trkseg, 'trkpt')
        trkpt.set('lat', f"{lat:.7f}")
        trkpt.set('lon', f"{lon:.7f}")

        # Calculate distance from previous point
        if i > 0:
            prev_lat, prev_lon = route_points[i-1]
            segment_distance = haversine_distance(prev_lat, prev_lon, lat, lon)
            cumulative_distance += segment_distance

        # Add timestamp (assuming 6 min/km pace for marathon)
        pace_per_meter = 0.36  # seconds per meter (6 min/km)
        time_offset = cumulative_distance * pace_per_meter
        point_time = start_time + timedelta(seconds=time_offset)

        time_elem = ET.SubElement(trkpt, 'time')
        time_elem.text = point_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Add real elevation data
        ele_elem = ET.SubElement(trkpt, 'ele')
        ele_elem.text = f"{elevations[i]:.1f}"

    # Write the new GPX file
    tree = ET.ElementTree(gpx_root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

    print(f"Converted GPX saved to: {output_file}")
    print(f"Total points: {len(route_points)}")

    return total_distance

if __name__ == "__main__":
    input_file = "Marathon_Course.gpx"
    output_file = "Marathon_Course_Fixed.gpx"

    # Set to False to use fallback elevations if API is slow/unavailable
    use_elevation_api = True

    try:
        print("Converting GPX file with elevation data...")
        distance = convert_route_to_track(input_file, output_file, use_elevation_api)
        print(f"\nConversion complete! Upload {output_file} to Stryd.")
        print(f"Course distance: {distance/1000:.2f} km ({distance/1609.34:.2f} miles)")
        print(f"Note: Set use_elevation_api=False in script if elevation lookup is too slow")
    except Exception as e:
        print(f"Error: {e}")
        print("If elevation API fails, try running again with use_elevation_api=False")