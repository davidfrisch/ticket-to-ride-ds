import cv2
from ultralytics import YOLO
import os
import json

def detect_cities(path_image: str):
    # Load image
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")

    # Load YOLO model
    model = YOLO("../models/cities_best_2.pt")
    results = model(image)

    # Extract filename for saving output
    filename = os.path.basename(path_image).split(".")[0]

    for i, result in enumerate(results):
        image_with_circles = image.copy()

        image_with_circles = get_cities(result, image_with_circles)

        output_dir = "./detect_cities"
        os.makedirs(output_dir, exist_ok=True)

        # Save the modified image
        output_path = os.path.join(output_dir, f"cities_{filename}.jpg")
        success = cv2.imwrite(output_path, image_with_circles)
        
        if not success:
            raise RuntimeError(f"Failed to save image: {output_path}")
        else:
            print(f"Saved: {output_path}")

def get_cities(result, image):
    boxes = result.boxes.xywhn  # Get normalized bounding boxes
    
    if boxes is None or len(boxes) == 0:
        return image

    height, width, _ = image.shape  # Get image dimensions
    cities = {}
    for i, box in enumerate(boxes.tolist()):  # Convert to list for safe unpacking
        x_center, y_center, w, h = box  # Normalized coordinates
        cities[i] = {"x": x_center, "y": y_center}
        # Convert to absolute pixel coordinates
        x_pixel = int(x_center * width)
        y_pixel = int(y_center * height)

        # Draw a circle at detected location
        cv2.circle(image, (x_pixel, y_pixel), 10, (0, 0, 255), -1)  # -1 fills the circle
        # add label
        cv2.putText(image, f"City {i}", (x_pixel, y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    with open("cities.json", "w") as f:
        json.dump(cities, f, indent=4)
        
    return image

if __name__ == '__main__':
    dir_path = "detect_map"
    for filename in os.listdir(dir_path):
        detect_cities(os.path.join(dir_path, filename))
