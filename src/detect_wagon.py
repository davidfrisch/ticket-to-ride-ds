from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("../models/wagons_1302.pt")

def detect_wagon(image_path):
    """Detect wagons in the given image."""
    # Load the image
    filename = image_path.split("/")[-1].split(".")[0]
    image = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Run inference
    results = model(image)

    # Process results
    for result in results:
        boxes = result.boxes  # Bounding box outputs
        masks = result.masks  # Segmentation masks
        keypoints = result.keypoints  # Pose keypoints
        probs = result.probs  # Classification probabilities
        obb = result.obb  # Oriented bounding boxes

        # Show and save the result
        result.show()
        result.save(filename=f"detect_shape/detect_wagons_{filename}.jpg")
    print("Detection complete.")
        
if __name__ == "__main__":
    detect_wagon("../output_folder_cropped/cropped_49_77_crop_0_1.jpg")