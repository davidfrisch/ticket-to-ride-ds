import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults
import os
import numpy as np
from typing import List, Optional
from collections import defaultdict
from City import City
from utils import draw_cities, identify_city_corners, get_cities_ref, rotate_image, get_angle_of_board
from math import sqrt
from constants import MAX_MISSING_CITIES

def get_points_normalized(result: UltralyticsResults) -> np.ndarray:
    """Extracts normalized bounding box centers from YOLO results."""
    boxes = result.boxes.xywhn  # Get normalized bounding boxes
    
    if boxes is None or len(boxes) == 0:
        return np.array([])

    normalized_pts = np.zeros((len(boxes), 2))
    
    for i, box in enumerate(boxes.tolist()):
        x_center, y_center, _, _ = box  
        normalized_pts[i] = (x_center, y_center)
        
    return normalized_pts


def avg_distance_between_points(points: np.ndarray) -> float:
    """Computes the average distance between points."""
    distances = []
    for i, point in enumerate(points):
        for j in range(i+1, len(points)):
            distance = np.linalg.norm(point - points[j])
            distances.append(distance)
            
    return np.mean(distances)


def remove_duplicates(cities: List[City], threshold=0.02) -> List[City]:
    """Removes duplicate cities that are too close together by averaging them."""
    unique_cities = []
    grouped_cities = defaultdict(list)

    # Group detected cities by name
    for city in cities:
        grouped_cities[city.name].append(city)

    # Process each city group
    lost_cities = []
    for city_name, instances in grouped_cities.items():
        if len(instances) == 1:
            unique_cities.append(instances[0])
            continue
        
        avg_distance = avg_distance_between_points(np.array([(city.x, city.y) for city in instances]))
        if avg_distance < threshold:
            x = np.mean([city.x for city in instances])
            y = np.mean([city.y for city in instances])
            unique_cities.append(City(name=city_name, x=x, y=y))
        else:
            lost_cities.append(city_name)

    return unique_cities



def find_missing_points(points_normalized: np.ndarray, ref_cities: List[City]) -> List[City]:
    ref_points = np.array([(city.x, city.y) for city in ref_cities])
    missing_points = []
    
    for ref_point in ref_points:
        found = False
        for point in points_normalized:
            distance = np.linalg.norm(ref_point - point)
            if distance < 0.02:
                found = True
                break
                
        if not found:
            missing_points.append(ref_point)
            
    return missing_points


def get_closest_cities(city_name: str, ref_cities: List[City]) -> List[City]:
    ref_points = np.array([(city.x, city.y) for city in ref_cities])
    city = next((c for c in ref_cities if c.name == city_name), None)
    
    if city is None:
        raise ValueError(f"City {city_name} not found.")
    
    distances = np.linalg.norm(ref_points - (city.x, city.y), axis=1)
    
    closest_indices = np.argsort(distances)
    
    closest_cities = []
    for i in closest_indices:
        if ref_cities[i].name != city_name:
            closest_cities.append(ref_cities[i])
  
        
    return closest_cities


def find_city(points_normalized: np.ndarray, city_name: str, ref_cities: List[City]) -> City:
    closest_cities = get_closest_cities(city_name, ref_cities)
    for city in closest_cities:
        distance = np.linalg.norm((city.x, city.y) - points_normalized, axis=1)
        if np.min(distance) < 0.02:
            return city
        
    return None
  
def find_closest_city(points_normalized: np.ndarray, ref_cities: List[City]) -> City:
    closest_city = None
    min_distance = np.inf
    for city in ref_cities:
        distance = sqrt((city.x - points_normalized[0]) ** 2 + (city.y - points_normalized[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_city = city
            
    return closest_city     


def get_missing_cities(cities: List[City], ref_cities: List[City]) -> List[str]:
    missing_cities = []
    for city in ref_cities:
        found = False
        for detected_city in cities:
            if city.name == detected_city.name:
                found = True
                break
                
        if not found:
            missing_cities.append(city.name)
            
    return missing_cities


def merge_close_pts(points_normalized: np.ndarray, threshold=0.02) -> np.ndarray:
    merged_pts = []
    for point in points_normalized:
        found = False
        for merged_point in merged_pts:
            distance = np.linalg.norm(point - merged_point)
            if distance < threshold:
                found = True
                break
                
        if not found:
            merged_pts.append(point)
            
    return np.array(merged_pts)


def find_points_on_board(image) -> np.ndarray:
    model = YOLO("../../models/cities_best_2.pt")
    results = model(image)
    
    if len(results) == 0:
        print("No board detected.")
        return np.array([])
    
    result = results[0]
    points_normalized = get_points_normalized(result)
    points_normalized = merge_close_pts(points_normalized)
    
    return points_normalized


def get_neighbours_cities(city_name: str, cities_on_board: List[City], ref_cities: List[City]) -> List[City]:
    city_connections = next((city.connections for city in ref_cities if city.name == city_name), None)
    neighbours = []
    for connection in city_connections:
        neighbour = next((city for city in cities_on_board if city.name == connection), None)
        if neighbour is not None:
            neighbours.append(neighbour)
            
    return neighbours




def find_city_based_on_other_cities(missing_city_name: str, cities_on_board: List[City], ref_cities: List[City]) -> Optional[City]:
    """Estimates the location of a missing city using relative distances from its neighbors."""
    print(f"Estimating position for {missing_city_name}")
    neighbours = get_neighbours_cities(missing_city_name, cities_on_board, ref_cities)
    if not neighbours:
        return None  # No known neighbors to infer position

    ref_city = next((city for city in ref_cities if city.name == missing_city_name), None)
    if not ref_city:
        print(f"Missing reference city for {missing_city_name}")
        return None  # Missing city not found in reference

    estimated_positions = []
    
    for neighbor in neighbours:
        ref_neighbor = next((c for c in ref_cities if c.name == neighbor.name), None)
        if not ref_neighbor:
            print(f"Missing reference city for {neighbor.name}")
            continue
        
        # Compute vector from neighbor to missing city in reference map
        dx = ref_city.x - ref_neighbor.x
        dy = ref_city.y - ref_neighbor.y

        # Apply same displacement in the actual board
        estimated_x = neighbor.x + dx
        estimated_y = neighbor.y + dy
        estimated_positions.append((estimated_x, estimated_y))

    if not estimated_positions:
        print(f"No estimated positions for {missing_city_name}")
        return None  # No valid position found

    # Compute best estimate (e.g., mean, median, or another function)
    estimated_x = sum(x for x, _ in estimated_positions) / len(estimated_positions)
    estimated_y = sum(y for _, y in estimated_positions) / len(estimated_positions)

    return City(name=missing_city_name, x=estimated_x, y=estimated_y, connections=[n.name for n in neighbours])
    
    
    
    


def find_cities(normalised_pts: np.ndarray) -> List[City]:
    ref_cities = get_cities_ref()
    ref_cities_by_x = sorted(ref_cities, key=lambda city: city.x)
    ref_cities_by_y = sorted(ref_cities, key=lambda city: city.y)
    
    # find city traversing the board from left to right
    cities = []
    for i, point in enumerate(normalised_pts):
        city = find_closest_city(point, ref_cities_by_x)
        cities.append(City(name=city.name, x=point[0], y=point[1]))
        
    # find city traversing the board from top to bottom
    for i, point in enumerate(normalised_pts):
        city = find_closest_city(point, ref_cities_by_y)
        cities.append(City(name=city.name, x=point[0], y=point[1]))
        
    cities = remove_duplicates(cities)
    missing_cities = get_missing_cities(cities, ref_cities)
    
    if len(missing_cities) > MAX_MISSING_CITIES:
        print("Too many missing cities, skipping estimation")
        return cities
      
      
    print(f"Missing cities: {[city for city in missing_cities]}")
    for missing_city in missing_cities.copy():
        city = find_city_based_on_other_cities(missing_city, cities, ref_cities)
        if city is not None:
            cities.append(city)
            missing_cities.remove(missing_city)
    
    print(f"Missing cities after estimation: {[city for city in missing_cities]}")
    return cities


def detect_cities(path_image: str):
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    ref_cities = get_cities_ref()
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")
    
    pts = find_points_on_board(image)
    num_missing_pts = len(ref_cities) - len(pts)
    print(f"Missing pts {num_missing_pts}")
    if num_missing_pts != 0:
        print("Missing points, skipping")
        return
    
    cities = find_cities(pts)
    
    os.makedirs("outputs", exist_ok=True)
    filename = path_image.split("/")[-1].split(".")[0]
    draw_cities(image, cities)
    cv2.imwrite(f"outputs/cities_{filename}.jpg", image)

    

if __name__ == '__main__':
    dir_path = "../detect_map"
    for filename in os.listdir(dir_path):
        if filename.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            continue
        print(f"Processing {filename}...")
        detect_cities(os.path.join(dir_path, filename))
