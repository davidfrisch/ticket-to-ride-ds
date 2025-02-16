from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("../models/wagons_1302.pt")


mask_red = ([ 17,  15, 100], [ 50,  56, 200])
mask_blue = ([100,  60,  15], [225, 135,  75])


def mask_color(image, lower, upper):
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)
  

def detect_wagons(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
      
    # Mask the image
    mask_red_image = mask_color(image, mask_red[0], mask_red[1])
    mask_blue_image = mask_color(image, mask_blue[0], mask_blue[1])
    
    cv2.imwrite("detect_wagons/detect_wagons_mask_red.jpg", mask_red_image)
    cv2.imwrite("detect_wagons/detect_wagons_mask_blue.jpg", mask_blue_image)
    
    






















# def detect_wagon(image_path):
#     """Detect wagons in the given image."""
#     # Load the image
#     filename = image_path.split("/")[-1].split(".")[0]
#     image = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")

#     # Run inference
#     results = model(image)

#     # Process results
#     for result in results:
#         boxes = result.boxes  # Bounding box outputs
#         masks = result.masks  # Segmentation masks
#         keypoints = result.keypoints  # Pose keypoints
#         probs = result.probs  # Classification probabilities
#         obb = result.obb  # Oriented bounding boxes

#         # Show and save the result
#         result.show()
#         result.save(filename=f"detect_wagons/wagons_{filename}.jpg")
#     print("Detection complete.")
        
if __name__ == "__main__":
    detect_wagons("../output_folder_cropped/cropped_49_77_crop_0_1.jpg")