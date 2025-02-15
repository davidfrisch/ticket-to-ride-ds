import cv2
from ultralytics import YOLO
import os
import json
import numpy as np
import pydantic
from typing import List
from collections import defaultdict


class City(pydantic.BaseModel):
    name: str
    x: float
    y: float

    def __str__(self):
        return f"{self.name}: ({self.x}, {self.y})"

    def __repr__(self):
        return str(self)


def get_cities_ref():
    with open("./data/cities.json", "r") as f:
        cities_names = json.load(f)
    if cities_names is None:
        raise FileNotFoundError("Could not read cities.json")
    return cities_names
      
  
def find_closest_city(x, y) -> str:
    """Finds the closest city from the reference map."""
    cities = get_cities_ref()
    closest_city = None
    min_distance = float("inf")
    
    for city, coords in cities.items():
        distance = (x - coords["x"])**2 + (y - coords["y"])**2
        if distance < min_distance:
            min_distance = distance
            closest_city = city
            
    return closest_city  


def distance_between_cities(city1: City, city2: City) -> float:
    """Calculates the Euclidean distance between two cities."""
    return ((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2) ** 0.5
  

def get_cities_normalized(result, image) -> np.ndarray:
    """Extracts normalized bounding box centers from YOLO results."""
    boxes = result.boxes.xywhn  # Get normalized bounding boxes
    
    if boxes is None or len(boxes) == 0:
        return np.array([])

    normalized_pts = np.zeros((len(boxes), 2))
    
    for i, box in enumerate(boxes.tolist()):
        x_center, y_center, _, _ = box  
        normalized_pts[i] = (x_center, y_center)
        
    return normalized_pts


def remove_duplicates(cities: List[City], threshold=0.1) -> List[City]:
    """Keeps only the closest detected city to the reference coordinates, ignoring outliers."""
    unique_cities = []
    grouped_cities = defaultdict(list)
    cities_ref = get_cities_ref()  # Reference city locations

    # Group detected cities by name
    for city in cities:
        grouped_cities[city.name].append(city)

    # Keep only the closest detection per city within the threshold
    for city_name, instances in grouped_cities.items():
        ref_coords = cities_ref.get(city_name)

        if ref_coords is None:
            # If no reference exists, just keep the first detected instance
            best_city = min(instances, key=lambda c: (c.x**2 + c.y**2))  
        else:
            ref_x, ref_y = ref_coords["x"], ref_coords["y"]
            
            # Find the closest detection
            best_city = min(instances, key=lambda c: (c.x - ref_x) ** 2 + (c.y - ref_y) ** 2)
            min_distance = ((best_city.x - ref_x) ** 2 + (best_city.y - ref_y) ** 2) ** 0.5  # Euclidean distance

            # Only keep if it's within the threshold
            print(f"Distance to {city_name}: {min_distance}")
            if min_distance > threshold:
                continue  # Ignore detections that are too far

        unique_cities.append(best_city)

    return unique_cities

 
def predict_missing_city(missing_city, known_cities):
    """
    Predict the missing city's location based on the nearest known cities.
    
    Args:
        missing_city (str): The name of the missing city to predict.
        known_cities (dict): A dictionary of known cities with their normalized coordinates.

    Returns:
        tuple: The predicted (x, y) normalized coordinates of the missing city.
    """
    # If the city is already in the known cities, no need to predict
    if missing_city in known_cities:
        print(f"{missing_city} is already detected.")
        return known_cities[missing_city]

    # For prediction, find the nearest cities from known cities (simple Euclidean distance)
    missing_city_coords = known_cities.get(missing_city)
    if not missing_city_coords:
        print(f"{missing_city} not found in the known cities list.")
        return None
    
    distances = []
    for city, coords in known_cities.items():
        if city == missing_city:
            continue
        distance = np.linalg.norm(np.array([missing_city_coords['x'], missing_city_coords['y']]) - np.array([coords['x'], coords['y']]))
        distances.append((city, distance, coords))
    
    # Sort by distance, pick the closest few cities to calculate the predicted position
    distances.sort(key=lambda x: x[1])  # Sort by distance

    # Choose top 3 nearest cities (you can adjust based on need)
    closest_cities = distances[:3]
    
    # Average their positions for the predicted city location
    avg_x = np.mean([city[2]['x'] for city in closest_cities])
    avg_y = np.mean([city[2]['y'] for city in closest_cities])

    print(f"Predicted location of {missing_city}: ({avg_x}, {avg_y})")
    return {'x': avg_x, 'y': avg_y}



def predict_cities(image) -> List[City]:
    model = YOLO("../models/cities_best_2.pt")
    results = model(image)

    if len(results) == 0:
        print("No cities detected.")
        return image
    
    result = results[0]
    image_with_circles = image.copy()
    cities_normalized = get_cities_normalized(result, image_with_circles)
    
    cities_predicted: List[City] = []
    for city_normalized in cities_normalized:
        x, y = city_normalized
        city = find_closest_city(x, y)
        cities_predicted.append(City(name=city, x=x, y=y))
        
    
    cities_filtered = remove_duplicates(cities_predicted)
    found_cities_names = [city.name for city in cities_filtered]
    known_cities = get_cities_ref()
    
    num_missing_cities = len(known_cities) - len(found_cities_names)
    if num_missing_cities > 0:
        print(f"Found {len(found_cities_names)} cities. {num_missing_cities} cities missing.")
        # for city in known_cities.keys():
        #     if city not in found_cities_names:
        #         print(f"City {city} is missing.")
        #         predicted_coords = predict_missing_city(city, known_cities)
        #         if predicted_coords:
        #             print(f"Predicted city {city}: {predicted_coords}")
        #             cities_filtered.append(City(name=city, x=predicted_coords['x'], y=predicted_coords['y']))
                    
                
    else:
        print("All cities detected.")
        
    return cities_filtered
    

    
        
        
def draw_cities(image, cities: List[City]):
    for city in cities:
        x, y = int(city.x * image.shape[1]), int(city.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image, city.name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    return image



def detect_cities(path_image: str):
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")
    
    cities = predict_cities(image)
    image_with_cities = draw_cities(image, cities)
    filename = path_image.split("/")[-1].split(".")[0]
    os.makedirs("detect_cities", exist_ok=True)
    
    image_with_cities = draw_cities(image, cities)
    cv2.imwrite(f"detect_cities/cities_{filename}.jpg", image_with_cities)

    
def draw_cities(image, cities: List[City]):
    for city in cities:
        x, y = int(city.x * image.shape[1]), int(city.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image, city.name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    return image
    
    

if __name__ == '__main__':
    dir_path = "detect_map"
    for filename in os.listdir(dir_path):
        detect_cities(os.path.join(dir_path, filename))
