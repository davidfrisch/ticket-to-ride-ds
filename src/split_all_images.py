import cv2
from ultralytics import YOLO
import os



def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]
  
def crop_and_save_image(image, x, y, w, h, output_folder):
    cropped_img = crop_image(image, x, y, w, h)
    cv2.imwrite(output_folder, cropped_img)


def rotate_image(image):
    # rotate such that the width is higher than the height
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image
    

def split_image(image, rows, cols, filename, output_folder):
    image = rotate_image(image)
    img_h, img_w, _ = image.shape  # Height, Width, Channels
    square_h = img_h // rows
    square_w = img_w // cols

    os.makedirs(output_folder, exist_ok=True)
    
    for row in range(rows):
        for col in range(cols):
            x1, y1 = col * square_w, row * square_h
            x2, y2 = x1 + square_w, y1 + square_h

            crop_filename = f"{output_folder}/{filename}_crop_{row}_{col}.jpg"
            crop_and_save_image(image, x1, y1, square_w, square_h, crop_filename)
            
            
# Load the image
board_img = cv2.imread("./cropped_49_77.jpg", cv2.IMREAD_UNCHANGED)

split_image(board_img, 3, 3, "cropped_49_77", "output_folder")


model = YOLO("./board_model.pt")

for filename in os.listdir("./boards")[:1]:
    img = cv2.imread(f"./boards/{filename}")
    results = model(img)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

        # crop the image
        for box in boxes:
            x1, y1, x2, y2 = box
            split_image(img, 3, 2, filename, "cropped_squares")
        

    
    