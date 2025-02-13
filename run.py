import cv2
import numpy as np

# Define the HSV range for green
lower_green = np.array([40, 50, 50])   # Lower bound (Hue, Saturation, Value)
upper_green = np.array([90, 255, 255]) # Upper bound

# Load the image
image = cv2.imread("./images/img_5.png")  # Change this to your image filename
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours (objects in the green range)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for contour in contours:
    if cv2.contourArea(contour) > 200:  # Ignore small noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Show the result
cv2.imshow("Green Wagon Tracking", image)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()