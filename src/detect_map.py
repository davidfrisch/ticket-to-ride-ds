import cv2
from ultralytics import YOLO
import os


def detect_map(path_image: str, output_dir="detect_map"):
    # Load image
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")

    # Load model
    model = YOLO("../models/board_model_1502.pt")

    # Rotate image if needed
    image_copy = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Run YOLO detection
    results = model(image_copy)

    # Handle cases where no or multiple objects are detected
    if len(results) == 0:
        print(f"No objects detected in {path_image}")
        return
    elif len(results) > 1:
        print(f"Multiple objects detected in {path_image}, skipping")
        return

    # Extract detected bounding box
    box = results[0].boxes[0]

    # Crop and save board
    board = crop_board(image_copy, box)
    filename = os.path.splitext(os.path.basename(path_image))[0]
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"board_{filename}.jpg"), board)


def crop_board(image, box):
    """Crop the board from the image using bounding box coordinates."""
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Ensure integer values
    return image[y1:y2, x1:x2]  # Proper slicing


if __name__ == '__main__':
    detect_map("../boards/IMG_9653.jpg")

    # Uncomment for batch processing
    # for filename in os.listdir("../boards"):
    #     if filename.lower().endswith((".jpg", ".png")):
    #         detect_map(f"../boards/{filename}")
