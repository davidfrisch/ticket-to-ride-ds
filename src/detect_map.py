import cv2
from ultralytics import YOLO
import os
def detect_map(path_image: str):
    image = cv2.imread(path_image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")
      
    model = YOLO("../models/board_model_1502.pt")
    results = model(image)
    if len(results) == 0:
        print("No objects detected")
        return
      
    if len(results) > 1:
        print("Multiple objects detected")
        return
            
    box = results[0].boxes[0]
    
    
    board = extract_board(image, box)
    filename = path_image.split("/")[-1].split(".")[0]
    cv2.imwrite(f"detect_map/board_{filename}.jpg", board)
    

def extract_board(image, box):
    x, y, w, h = box.xyxy[0]
    return image[int(y):int(y+h), int(x):int(x+w)]
  


if __name__ == '__main__':
    detect_map("../boards/board_1652.jpeg")
    # for filename in os.listdir("../boards"):
    #     detect_map(f"../boards/{filename}")