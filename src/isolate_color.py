import cv2
import numpy as np

# Define the HSV range for different colors
COLOR_RANGES = {
    "green":  (np.array([40, 50, 50]), np.array([90, 255, 255])),
    "red":    [(np.array([0, 100, 50]), np.array([10, 255, 255])),  # Two ranges for red
               (np.array([170, 100, 50]), np.array([180, 255, 255]))],
    "black":  (np.array([0, 0, 0]), np.array([180, 255, 50])),
    "yellow": (np.array([20, 100, 100]), np.array([35, 255, 255])),
    "blue":   (np.array([100, 100, 50]), np.array([130, 255, 255]))
}


def isolate_color(image, color):
    """Create a binary mask for the given color."""
    if color not in COLOR_RANGES:
        raise ValueError(f"Color '{color}' not defined in COLOR_RANGES.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_range = COLOR_RANGES[color]

    if color == "red":
        # If the color has two ranges (like red), merge the masks
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
    """Find and return contours for the given color."""
    mask = isolate_color(image, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > min_area], mask


def draw_contours(image, contours, color_name):
    """Draw bounding boxes around detected objects of the specified color."""
    color_bgr = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "black": (0, 0, 0),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
    }
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color_bgr.get(color_name, (255, 255, 255)), 3)

    return image


def main(image_path, track_color):
    """Main function to detect and track a specific color."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    contours, mask = find_contours(image, track_color)

    # Draw bounding boxes on the detected objects
    result_image = draw_contours(image, contours, track_color)

    # Save the result image
    cv2.imwrite("result.jpg", result_image)


if __name__ == "__main__":
    # Change the file path and color name as needed
    main("output_folder_cropped/cropped_49_77_crop_0_0.jpg", "yellow")  # Example: Tracking green wagons
