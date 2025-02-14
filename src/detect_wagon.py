from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("./wagons_1302.pt")

# Read the image in grayscale
image_gray = cv2.imread("output_folder_cropped/cropped_49_77_crop_0_0.jpg", cv2.IMREAD_GRAYSCALE)

# Ensure image is loaded
if image_gray is None:
    raise FileNotFoundError("Error: Could not load image. Check the file path.")

# Convert grayscale to 3-channel format
image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

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
    result.save(filename="result.jpg")
