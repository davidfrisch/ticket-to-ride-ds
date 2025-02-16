import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults
import os
import numpy as np
from typing import List, Optional
from collections import defaultdict
from City import City
from utils import draw_cities, identify_city_corners, get_cities_ref, rotate_image, get_angle_of_board, distance_between_points
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
    min_distance = 0.05
    possible_cities = []
    for city in ref_cities:
        distance = sqrt((city.x - points_normalized[0]) ** 2 + (city.y - points_normalized[1]) ** 2)
            
        if distance < min_distance:
            possible_cities.append(city)
    
    
    if len(possible_cities) == 1:
        return City(name=possible_cities[0].name, x=points_normalized[0], y=points_normalized[1], connections=possible_cities[0].connections)

    elif len(possible_cities) > 1:
        print(f"Multiple cities found for point {points_normalized}")
        return None
      
    return None  


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




def find_city_based_on_other_cities(missing_city_name: str, pts_with_no_city: np.ndarray, cities_on_board: List[City], ref_cities: List[City]) -> Optional[City]:
    """Estimates the location of a missing city using reference neighbors and matches it to the closest unidentified point."""
    print(f"Estimating position for {missing_city_name}")
    neighbours = get_neighbours_cities(missing_city_name, cities_on_board, ref_cities)
    if not neighbours:
        return None  # No known neighbors to infer position

    ref_city = next((city for city in ref_cities if city.name == missing_city_name), None)
    if not ref_city:
        return None  # Missing city not found in reference

    estimated_positions = []
    
    for neighbor in neighbours:
        ref_neighbor = next((c for c in ref_cities if c.name == neighbor.name), None)
        if not ref_neighbor:
            continue
        
        # Compute vector from neighbor to missing city in reference map
        dx = ref_city.x - ref_neighbor.x
        dy = ref_city.y - ref_neighbor.y

        # Apply same displacement in the actual board
        estimated_x = neighbor.x + dx
        estimated_y = neighbor.y + dy
        estimated_positions.append((estimated_x, estimated_y))

    if not estimated_positions:
        return None  # No valid estimate

    # Compute the best estimated position (mean or weighted approach)
    estimated_x = sum(x for x, _ in estimated_positions) / len(estimated_positions)
    estimated_y = sum(y for _, y in estimated_positions) / len(estimated_positions)
    estimated_position = np.array([estimated_x, estimated_y])

    # Find the closest point in `pts_with_no_city`
    distances = [distance_between_points(estimated_position, (point.x, point.y)) for point in pts_with_no_city]
    closest_point = pts_with_no_city[np.argmin(distances)]
    return City(name=missing_city_name, x=closest_point.x, y=closest_point.y, connections=ref_city.connections)
    
    
    


def find_cities(normalised_pts: np.ndarray) -> List[City]:
    ref_cities = get_cities_ref()
    cities = []
  
    for point in normalised_pts:
        city = find_closest_city(point, ref_cities)
        if city is not None:
            cities.append(city)
        
    cities = remove_duplicates(cities)
    
    pts_with_no_city = [City(name="UNKNOWN", x=point[0], y=point[1]) for point in normalised_pts if not any(city.x == point[0] and city.y == point[1] for city in cities)]
    
    missing_cities = get_missing_cities(cities, ref_cities)
    if len(missing_cities) > MAX_MISSING_CITIES:
        print(f"Too many missing cities: {len(missing_cities)}")
        return cities
      
    print(f"Missing cities: {[city for city in missing_cities]}")
    for missing_city in missing_cities.copy():
        city = find_city_based_on_other_cities(missing_city, pts_with_no_city, cities, ref_cities)
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
