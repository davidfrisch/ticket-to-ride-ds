import cv2

mask_red =  ([ 38,  50,  50], [ 75, 255, 255])
mask_blue = ([ 90,  50,  50], [130, 255, 255])
mask_green = ([ 38,  50,  50], [ 75, 255, 255])
mask_yellow =  ([ 38,  50,  50], [ 75, 255, 255])
mask_black = ([ 0,  0,  0], [ 179, 255, 50])


def mask_color(image, lower, upper):
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)
  

def detect_wagons_with_mask(image_path):
    filename = image_path.split("/")[-1].split(".")[0]
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
      
    # Mask the image
    mask_red_image = mask_color(image, mask_red[0], mask_red[1])
    mask_blue_image = mask_color(image, mask_blue[0], mask_blue[1])
    mask_green_image = mask_color(image, mask_green[0], mask_green[1])
    mask_yellow_image = mask_color(image, mask_yellow[0], mask_yellow[1])
    mask_black_image = mask_color(image, mask_black[0], mask_black[1])
    
    cv2.imwrite(f"outputs/masks/red_{filename}.jpg", mask_red_image)
    cv2.imwrite(f"outputs/masks/blue_{filename}.jpg", mask_blue_image)
    cv2.imwrite(f"outputs/masks/green_{filename}.jpg", mask_green_image)
    cv2.imwrite(f"outputs/masks/yellow_{filename}.jpg", mask_yellow_image)
    cv2.imwrite(f"outputs/masks/black_{filename}.jpg", mask_black_image)

# def _detect_wagons(image_path):
#     """Detect wagons in the given image."""
#     # Load the image
#     filename = image_path.split("/")[-1].split(".")[0]
#     image = cv2.imread(image_path, cv2.COLOR_GRAY2BGR)
#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")

#     results = model(image)

#     # Process results
#     for result in results:
#         boxes = result.boxes  # Bounding box outputs
#         masks = result.masks  # Segmentation masks
#         keypoints = result.keypoints  # Pose keypoints
#         probs = result.probs  # Classification probabilities
#         obb = result.obb  # Oriented bounding boxes


#         result.save(filename=f"outputs/wagons_{filename}.jpg")
#     print("Detection complete.")


# def put_image_in_bw(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Could not read image: {image_path}")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(image_path, image)
