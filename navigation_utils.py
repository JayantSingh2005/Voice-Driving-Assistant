# navigation_utils.py
import requests
import os

ORS_API_KEY = os.getenv("ORS_API_KEY") or "5b3ce3597851110001cf62485e7fb6199d944021b8aab717a7b91e60"
ORS_BASE_URL = "https://api.openrouteservice.org"

def geocode_place(place):
    """
    Geocodes a place name to its (latitude, longitude) coordinates using OpenRouteService.
    """
    url = f"{ORS_BASE_URL}/geocode/search"
    params = {"api_key": ORS_API_KEY, "text": place, "size": 1}
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.ok and res.json().get("features"):
            coords = res.json()["features"][0]["geometry"]["coordinates"]
            return coords[1], coords[0]  # ORS returns [lon, lat], we want (lat, lon)
    except requests.RequestException as e:
        print(f"Error geocoding place '{place}': {e}")
        return None
    return None

def get_route_details(start_coords: tuple, end_coords: tuple):
    """
    Fetches route details, including summary and turn-by-turn instructions,
    using OpenRouteService.

    Args:
        start_coords (tuple): (latitude, longitude) of the starting point.
        end_coords (tuple): (latitude, longitude) of the destination point.

    Returns:
        dict: A dictionary containing route summary, turn-by-turn steps, and geometry,
              or None if the route cannot be calculated.
    """
    url = f"{ORS_BASE_URL}/v2/directions/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {
        "coordinates": [[start_coords[1], start_coords[0]], [end_coords[1], end_coords[0]]],
        "instructions": True,  # Request turn-by-turn instructions
        "instructions_format": "text", # "text" or "html"
        "geometry": True # Request geometry for map rendering
    }
    try:
        res = requests.post(url, json=body, headers=headers, timeout=10)
        res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = res.json()

        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            summary = route["summary"]
            
            # Extract turn-by-turn instructions
            instructions = []
            if route.get("segments"):
                for segment in route["segments"]:
                    for step in segment.get("steps", []):
                        instructions.append({
                            "instruction": step.get("instruction"),
                            "distance_meters": step.get("distance"),
                            "duration_seconds": step.get("duration"),
                            "name": step.get("name"), # Road name
                            "type": step.get("type"), # Maneuver type (e.g., "turn", "depart", "arrive")
                            "way_points": step.get("way_points") # [start_index, end_index] in the geometry coordinates
                        })

            return {
                "distance_km": round(summary["distance"] / 1000, 2),
                "duration_min": round(summary["duration"] / 60, 2),
                "turn_by_turn_steps": instructions,
                "geometry": route.get("geometry") # GeoJSON linestring for rendering
            }
    except requests.RequestException as e:
        print(f"Error getting route details: {e}")
        return None
    return None