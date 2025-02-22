from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from ultralytics.utils import ops

# Load YOLO model
model = YOLO("../../models/wagons_2002.pt")


def angle_between_points(p1, p2) -> float:
    angle = np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0]) * 180 / np.pi
    return angle


def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def rotate_point(point, angle, center):
    angle = np.radians(angle)
    translated = point - center
    rotated = np.array([
        translated[0] * np.cos(angle) - translated[1] * np.sin(angle),
        translated[0] * np.sin(angle) + translated[1] * np.cos(angle)
    ])
    return rotated + center


def crop_image(image, pt1, pt2, curve, padding=20):
    height, width = image.shape[:2]

    # Compute bounding box
    line_length = np.linalg.norm(np.array(pt2) - np.array(pt1))
    
    x1 = min(pt1[0], pt2[0]) - line_length // 2
    y1 = min(pt1[1], pt2[1]) - padding - abs(curve)
    x2 = max(pt1[0], pt2[0]) + line_length // 2
    y2 = max(pt1[1], pt2[1]) + padding + abs(curve)

    if y2 - y1 < 100:
        y1 -= 50
        y2 += 50
        
    if abs(y2 - y1 - ((y2 - y1) / 2)) < 100:
        y1 -= abs(curve)
        y2 += abs(curve)
    
    # Ensure coordinates stay within image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))
    
    new_pt1 = (pt1[0] - x1, pt1[1] - y1)
    new_pt2 = (pt2[0] - x1, pt2[1] - y1)

    return image[y1:y2, x1:x2], new_pt1, new_pt2



def zoom_image(image, pt1, pt2):
  
    # Compute the zoom factor based on the line length
    width_image = image.shape[1]
    height_image = image.shape[0]
    zoom_factor = line_length(pt1, pt2) / 120
  
    height, width = image.shape[:2]
    
    # Resize the image
    zoomed_image = cv2.resize(image, (int(width * zoom_factor), int(height * zoom_factor)), interpolation=cv2.INTER_CUBIC)
    
    if zoom_factor > 1:  # Zooming in (crop center)
        new_h, new_w = zoomed_image.shape[:2]
        start_x = (new_w - width) // 2
        start_y = (new_h - height) // 2
        return zoomed_image[start_y:start_y + height, start_x:start_x + width]
    
    elif zoom_factor < 1:  # Zooming out (pad to keep original size)
        pad_h = (height - zoomed_image.shape[0]) // 2
        pad_w = (width - zoomed_image.shape[1]) // 2
        zoomed_image = cv2.copyMakeBorder(
            zoomed_image, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]  # Black padding
        )
        return zoomed_image[:height, :width]  # Ensure correct dimensions
    
    return zoomed_image

def save_per_route(image, filename, folder_name="./outputs/routes"):
    os.makedirs(folder_name, exist_ok=True)
    cv2.imwrite(f"{folder_name}/{filename}.jpg", image)


def normalised_to_pixel_coordinates(x, y, image_shape):
    return int(x * image_shape[1]), int(y * image_shape[0])


def load_cities_coordinates(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    folder_path = os.path.dirname(image_path)
    image = cv2.imread(image_path)
    
    with open(os.path.join(folder_path, f"{filename}.json"), "r") as file:
        data = json.load(file)
    
    cities = {}
    for city, coordinates in data.items():
        x_pixel, y_pixel = normalised_to_pixel_coordinates(coordinates["x"], coordinates["y"], image.shape)
        cities[city] = {"x": x_pixel, "y": y_pixel, "connections": coordinates["connections"]}
    
    return cities

def load_connections_info():
    with open("../data/connections.json", "r") as file:
        return json.load(file)    

connection_info = load_connections_info()
def find_connection_info(city_1, city_2):
    for connection in connection_info:
        if city_1 in connection and city_2 in connection:
            return connection_info[connection]
    return None


def line_length(pt1, pt2):
    return np.linalg.norm(pt2 - pt1)

def draw_curve(image, pt1, pt2, bend, color=(0, 255, 0), thickness=2):
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)

    # Compute the midpoint
    midpoint = (pt1 + pt2) / 2.0

    if bend == 0:
        # If bend is 0, draw a straight line
        return cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness), midpoint

    # Compute the perpendicular vector (90-degree rotation)
    direction = pt2 - pt1
    length = np.linalg.norm(direction)
    if length == 0:
        return image, midpoint

    perp_direction = np.array([-direction[1], direction[0]]) / length  # Normalize

    # Scale the perpendicular vector proportionally to the segment length
    arc_height = (length / 2) * (bend / 50)  # Adjust denominator to control sensitivity
    pt3 = midpoint + perp_direction * arc_height

    # Convert points to integers for drawing
    pt1 = tuple(pt1.astype(int))
    pt2 = tuple(pt2.astype(int))
    pt3 = tuple(pt3.astype(int))

    # Draw lines connecting pt1, pt3, and pt2
    cv2.line(image, pt1, pt3, color, thickness)
    cv2.line(image, pt3, pt2, color, thickness)

    return image, pt3
  
  
  
def is_intersection(box_xyxy, pt1, pt2, pt3):
    """
    Check if the bounding box (box_xyxy) intersects with either of the two lines:
    - Line 1: pt1 to pt2
    - Line 2: pt1 to pt3
    - Line 3: pt2 to pt3

    Args:
        box_xyxy (list): Bounding box in (x1, y1, x2, y2) format.
        pt1 (tuple): First point (x, y).
        pt2 (tuple): Second point (x, y).
        pt3 (tuple): Third point (x, y).

    Returns:
        bool: True if the bounding box intersects with any of the lines, else False.
    """
    x1, y1, x2, y2 = box_xyxy  # Extract box coordinates
    
    # Define the bounding box edges as line segments
    box_lines = [
        ((x1, y1), (x2, y1)),  # Top edge
        ((x2, y1), (x2, y2)),  # Right edge
        ((x2, y2), (x1, y2)),  # Bottom edge
        ((x1, y2), (x1, y1))   # Left edge
    ]
    
    # Define the input lines to check for intersection
    lines_to_check = [(pt1, pt2), (pt1, pt3), (pt2, pt3)]
    
    # Check intersection for each line of the box with each of the two lines
    for box_line in box_lines:
        for line in lines_to_check:
            if line_intersects(box_line[0], box_line[1], line[0], line[1]):
                return True
    
    return False

def line_intersects(p1, p2, q1, q2):
    """
    Check if two line segments (p1, p2) and (q1, q2) intersect using cross products.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2) 


def main(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    cities_coordinates = load_cities_coordinates(image_path)
    image = cv2.imread(image_path)
    saved_routes = []

    for city, coordinates in cities_coordinates.items():
        for connection in coordinates["connections"]:
              
            looking_for = "WILNO_RICA"
            if city not in looking_for or connection not in looking_for:
                continue
              
            if (city, connection) in saved_routes or (connection, city) in saved_routes:
                continue
            
            connection_info = find_connection_info(city, connection)
            connection_coordinates = cities_coordinates[connection]
            
            pt1 = np.array([coordinates["x"], coordinates["y"]], dtype=np.float32)
            pt2 = np.array([connection_coordinates["x"], connection_coordinates["y"]], dtype=np.float32)
            
            center = (pt1 + pt2) / 2  # Rotate around the midpoint
            angle = angle_between_points(pt1, pt2)
            rotated_image = rotate_image(image.copy(), -angle, tuple(center))
        
            
            pt1_rotated = rotate_point(pt1, angle, center)
            pt2_rotated = rotate_point(pt2, angle, center)
            
            
            curve = connection_info["curve"] if connection_info["curve"] is not None else 0
            cropped_image, crop_pt1, crop_pt2 = crop_image(rotated_image, pt1_rotated, pt2_rotated, curve)
            _, pt3 = draw_curve(rotated_image, crop_pt1, crop_pt2, curve)
            
            
            if cropped_image.size == 0:
                print(f"Error: {city} - {connection} image size is 0")
                continue
            
            
            saved_routes.append((city, connection))
            results = model(cropped_image)
            img_with_boxes = results[0].plot(labels=None)
            for box in results[0].boxes.xywh:
                box_xyxy = ops.xywh2xyxy(box.unsqueeze(0))[0].tolist() 
                if is_intersection(box_xyxy, crop_pt1, crop_pt2, pt3):
                    print(f"Intersection found between {city} and {connection}")

            image_with_line, _ = draw_curve(img_with_boxes.copy(), crop_pt1, crop_pt2, curve)
            cv2.imwrite(f"./outputs/routes/{filename}_{city}_{connection}.jpg", image_with_line)        
  
if __name__ == "__main__":
    main("../detect_cities/outputs/cities_board_img_1.jpg")
