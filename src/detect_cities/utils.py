from City import City
from typing import List
import cv2
from constants import TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, MIDDLE_LEFT, MIDDLE_DOWN
import json
import numpy as np

def distance_between_cities(city1: City, city2: City) -> float:
    """Calculates the Euclidean distance between two cities."""
    return ((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2) ** 0.5

def distance_between_points(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_cities_ref() -> dict[str, City]:
    with open("../data/cities.json", "r") as f:
        cities_json = json.load(f)

    cities: List[City] = {}
    for city, attributes in cities_json.items():
        cities[city] = City(name=city, x=attributes["x"], y=attributes["y"], connections=attributes.get("connections", []))
    
    return list(cities.values())
    

def rotate_image(image, angle):
    """ Rotates the image by the given angle while keeping full content visible """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Compute new bounding size
    cos, sin = np.abs(rot_mat[0, 0]), np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Rotate image
    rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    
    return rotated
 
 
def get_angle_of_board(points_normalized):
    """
    Compute the angle needed to rotate the board so that the 3 lowest points are parallel to the x-axis.
    """
    # Sort points by y value (descending, so lowest points come last)
    points_sorted = sorted(points_normalized, key=lambda p: p[1], reverse=True)

    # Select the 3 lowest points
    lowest_points = points_sorted[:2]  # Last two points with highest y-values

    # Extract x and y coordinates
    x_coords, y_coords = zip(*lowest_points)

    # Fit a line to these points (1st-degree polynomial)
    slope, intercept = np.polyfit(x_coords, y_coords, 1)

    # Compute the angle with respect to the x-axis
    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


  
def draw_cities(image, cities: List[City]):
    for city in cities:
        x, y = int(city.x * image.shape[1]), int(city.y * image.shape[0])
        color = (0, 255, 0)
        radius_relative = 0.01
        radius = int(min(image.shape[0], image.shape[1]) * radius_relative)
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.putText(image, city.name, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
        
    return image


def closest_city_corner(corner, points_normalized):
    """Finds the closest city to the given coordinates."""
    ref_cities = get_cities_ref()
    city_name = corner["name"]
    coords = corner["coords"]
    
    min_distance = float("inf")
    closest_point = None
    
    for point in points_normalized:
        x, y = point
        distance = ((x - coords[0]) ** 2 + (y - coords[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_point = point
            
    if closest_point is None:
        return None
            
    return City(name=city_name, x=closest_point[0], y=closest_point[1], connections=ref_cities[city_name].connections)
  


def identify_city_corners(points_normalized) -> List[City]:
    corners = []
    ref_cities = get_cities_ref()
    most_top_left = closest_city_corner(TOP_LEFT, points_normalized)
    if most_top_left is not None:
        corners.append(City(name="EDINBURG", x=most_top_left.x, y=most_top_left.y, connections=ref_cities["EDINBURG"].connections))
        
    most_top_right = closest_city_corner(TOP_RIGHT, points_normalized)
    if most_top_right is not None:
        corners.append(City(name="CONSTANTINOPLE", x=most_top_right.x, y=most_top_right.y, connections=ref_cities["CONSTANTINOPLE"].connections))
        
    most_bottom_left = closest_city_corner(BOTTOM_LEFT, points_normalized)
    if most_bottom_left is not None:
        corners.append(City(name="LISBOA", x=most_bottom_left.x, y=most_bottom_left.y, connections=ref_cities["LISBOA"].connections))
        
    most_bottom_right = closest_city_corner(BOTTOM_RIGHT, points_normalized)
    if most_bottom_right is not None:
        corners.append(City(name="ERZURUM", x=most_bottom_right.x, y=most_bottom_right.y, connections=ref_cities["ERZURUM"].connections))
    
    most_middle_left = closest_city_corner(MIDDLE_LEFT, points_normalized)
    if most_middle_left is not None:
        corners.append(City(name="BREST", x=most_middle_left.x, y=most_middle_left.y, connections=ref_cities["BREST"].connections))
    
    most_middle_down = closest_city_corner(MIDDLE_DOWN, points_normalized)
    if most_middle_down is not None:
        corners.append(City(name="PALERMO", x=most_middle_down.x, y=most_middle_down.y, connections=ref_cities["PALERMO"].connections))
  
    return corners


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
