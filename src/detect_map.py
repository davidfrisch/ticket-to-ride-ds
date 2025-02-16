import cv2
from ultralytics import YOLO
import os


def find_cls_index(result, cls_name):
    for index, name in result.names.items():
        if name == cls_name:
            return index
    return -1
  
def take_higest_confidence(result, cls_index):
    max_conf = 0
    max_conf_index = 0
    for i, box in enumerate(result.boxes):
        if box.cls == cls_index and box.conf > max_conf:
            max_conf = box.conf
            max_conf_index = i
            
    return max_conf_index



    
def rotate_map(image, map_box, boat_box):
    # Get centers of the boat and map
    x_boat, y_boat, _, _ = map(int, boat_box.xywh[0])
    x_map, y_map, _, _ = map(int, map_box.xywh[0])

    h, w = image.shape[:2]

    # Determine boat's position relative to the map
    relative_x = "left" if x_boat < x_map else "right"
    relative_y = "above" if y_boat < y_map else "below"
    
    print(f"Boat is {relative_y} and {relative_x} relative to the map.")

    # Determine rotation based on relative position
    if relative_x == "left" and relative_y == "below":
        print("Bottom-left (correct position)")
        rotation = None  # Correct position
    elif relative_x == "right" and relative_y == "below":
        print("Bottom-right")
        rotation = cv2.ROTATE_90_CLOCKWISE  # Rotate 90° CW
    elif relative_x == "left" and relative_y == "above":
        print("Top-left")
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE  # Rotate 90° CCW
    else:
        print("Top-right")
        rotation = cv2.ROTATE_180  # Rotate 180°

    # Apply rotation if needed
    if isinstance(rotation, int):
        print(f"Rotating image {rotation}")
        image = cv2.rotate(image, rotation)

    return image
    
    
   

def detect_map(path_image: str, output_dir="detect_map"):
    # Load image
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")

    # Load model
    model = YOLO("../models/map_boat_best.pt")

    # Run YOLO detection
    
    results = model(image)

    results[0].save(filename="output.jpg")
    # Handle cases where no or multiple objects are detected
    if len(results) == 0:
        print(f"No objects detected in {path_image}")
        return
    elif len(results) > 1:
        print(f"Multiple objects detected in {path_image}, skipping")
        return

    # Extract detected bounding box
    boat_cls_index = find_cls_index(results[0], "boat")
    map_cls_index = find_cls_index(results[0], "map")
    
    if boat_cls_index == -1 or map_cls_index == -1:
        print(f"Could not find boat or map class in {path_image}")
        return
    
    
    boat_index = take_higest_confidence(results[0], boat_cls_index)
    map_index = take_higest_confidence(results[0], map_cls_index)
    
    boat_box = results[0].boxes[boat_index]
    map_box = results[0].boxes[map_index]
    
    # Crop and save board
    board = crop_board(image, map_box)
    board_copy = board.copy()
    
    board_copy = rotate_map(board_copy, map_box, boat_box)
        
    filename = os.path.splitext(os.path.basename(path_image))[0]
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"board_{filename}.jpg"), board_copy)


def crop_board(image, box):
    """Crop the board from the image using bounding box coordinates."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Ensure integer values
    return image[y1:y2, x1:x2]  # Proper slicing


if __name__ == '__main__':
    detect_map("../boards/europe/IMG_9618.jpg")
    # for filename in os.listdir("../boards/europe")[0::10]:
    #     if filename.lower().endswith((".jpg", ".png")):
    #         detect_map(f"../boards/europe/{filename}")
