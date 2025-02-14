from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("./wagons_1302.pt")

# Define the HSV range for different colors
COLOR_RANGES = {
    "green": (np.array([40, 50, 50]), np.array([90, 255, 255])),
    "red": [(np.array([0, 100, 50]), np.array([10, 255, 255])),
            (np.array([170, 100, 50]), np.array([180, 255, 255]))],
    "black": (np.array([0, 0, 0]), np.array([180, 255, 50])),
    "yellow": (np.array([20, 100, 100]), np.array([35, 255, 255])),
    "blue": (np.array([100, 100, 50]), np.array([130, 255, 255]))
}

def isolate_color(image, color):
    """Create a binary mask for the given color."""
    if color not in COLOR_RANGES:
        raise ValueError(f"Color '{color}' not defined in COLOR_RANGES.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_range = COLOR_RANGES[color]

    if color == "red":
        lower1, upper1 = color_range[0]
        lower2, upper2 = color_range[1]
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = color_range
        mask = cv2.inRange(hsv, lower, upper)

    return mask

def find_contours(image, color, min_area=200):
    """Find bounding boxes for a given color."""
    mask = isolate_color(image, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
    return boxes, mask

def do_boxes_intersect(box1, box2):
    """Check if two bounding boxes intersect."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def main(image_path, track_color):
    """Detect wagons using YOLO and filter by color-based intersection."""
    # Load image in grayscale and convert to 3-channel
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    
    # 1️⃣ **Run YOLO detection**
    results = model(image)
    yolo_boxes = [box.xyxy[0].tolist() for result in results for box in result.boxes]

    # Draw YOLO detections
    image_yolo = image.copy()
    for x1, y1, x2, y2 in yolo_boxes:
        cv2.rectangle(image_yolo, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)  # Green box

    # cv2.imwrite("yolo_detections.jpg", image_yolo)  # Save YOLO output

    # 2️⃣ **Run color detection**
    color_boxes, mask = find_contours(image, track_color)

    # Draw color-based detections
    image_color = image.copy()
    for x, y, w, h in color_boxes:
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Blue box

    cv2.imwrite("color_detections.jpg", image_color)  # Save color output
    cv2.imwrite("color_mask.jpg", mask)  # Save color mask image

    # 3️⃣ **Filter YOLO boxes where intersection with color boxes occurs**
    selected_boxes = [box for box in yolo_boxes if any(do_boxes_intersect(box, c_box) for c_box in color_boxes)]

    # Draw final filtered bounding boxes
    image_final = image.copy()
    for x1, y1, x2, y2 in selected_boxes:
        cv2.rectangle(image_final, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # Red box for final selection

    cv2.imwrite("filtered_wagons.jpg", image_final)  # Save final result
 

if __name__ == "__main__":
    # Example: Detect yellow wagons
    main("output_folder_cropped/cropped_49_77_crop_0_0.jpg", "yellow")
