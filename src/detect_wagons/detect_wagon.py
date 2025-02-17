from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

# Load YOLO model
model = YOLO("../../models/wagons_1302.pt")


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


def crop_image(image, pt1, pt2, padding_x=200, padding_y=250):
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    x1, y1 = max(int(x1 - padding_x), 0), max(int(y1 - padding_y), 0)
    x2, y2 = min(int(x2 + padding_x), image.shape[1]), min(int(y2 + padding_y), image.shape[0])
    return image[y1:y2, x1:x2]


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


def main(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    cities_coordinates = load_cities_coordinates(image_path)
    image = cv2.imread(image_path)
    saved_routes = []

    for city, coordinates in cities_coordinates.items():
        for connection in coordinates["connections"]:
            if (city, connection) in saved_routes or (connection, city) in saved_routes:
                continue
            
            pt1 = np.array([coordinates["x"], coordinates["y"]], dtype=np.float32)
            connection_coordinates = cities_coordinates[connection]
            pt2 = np.array([connection_coordinates["x"], connection_coordinates["y"]], dtype=np.float32)
            
            angle = angle_between_points(pt1, pt2)
            center = (pt1 + pt2) / 2  # Rotate around the midpoint
            rotated_image = rotate_image(image.copy(), angle, tuple(center))

            
            pt1_rotated = rotate_point(pt1, -angle, center)
            pt2_rotated = rotate_point(pt2, -angle, center)
            
       
            cropped_image = crop_image(rotated_image, pt1_rotated, pt2_rotated)
            
            if cropped_image.size == 0:
                print(f"Error: {city} - {connection} image size is 0")
                continue
            
            
            saved_routes.append((city, connection))
    
    save_connections_to_json(saved_routes, filename)


def save_connections_to_json(connections, filename):
    json_connections = {}
    for connection in connections:
        city1, city2 = connection
        connection_name = f"{city1}_{city2}"
        json_connections[connection_name] = {
            "city1": city1,
            "city2": city2,
            "carriage": "",
            "color":	"",
            "tunnel": "",
            "engine": "",
          
        }
  
    with open(f"../data/connections_{filename}.json", "w") as file:
        json.dump(json_connections, file)


if __name__ == "__main__":
    main("../detect_cities/outputs/cities_board_IMG_9618.jpg")
