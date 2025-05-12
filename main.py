import logging
import os

import cv2
import numpy as np
from bioformats import ImageReader


def load_image(image_path):
    """
    Loads the image from the given path, handling .vsi format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: The image as a NumPy array, or None on error.
    """
    try:
        if image_path.lower().endswith(".vsi"):
            reader = ImageReader(image_path)
            # Assuming you want the first series and first channel.
            image = reader.read(series=0, c=0)
            reader.close()  # Close the reader to release resources
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(
                    f"Error: Could not read image at {image_path}"
                )
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    return image


def apply_contours(image):
    """
    Applies contour detection to the image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the original image and the contours found,
               or (None, None) on error.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return image, contours  # Return both image and contours
    except Exception as e:
        print(f"Error applying contours: {e}")
        return None, None


def perform_counts(image, contours):
    """
    Performs cell counting based on the detected contours.

    Args:
        image (numpy.ndarray): the original image
        contours (list): List of contours detected in the image.

    Returns:
        dict: A dictionary containing the counts for each cell type,
              or None on error.
    """
    if contours is None or image is None:
        return {"Cell Type 1": 0, "Cell Type 2": 0}  # Or error

    cell_type_1_count = 0
    cell_type_2_count = 0
    counted_image = image.copy()  # Create a copy to draw on

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (
            4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        )
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
        aspect_ratio = float(w) / h  # Calculate aspect ratio

        logging.info(f"(x, y, w, h): ({x}, {y}, {w}, {h})")
        logging.info(f"Aspect Ratio: {aspect_ratio}")

        # In real scenario, calculate mean_intensity for each stain.
        mean_intensity_1 = 100  # Dummy Value
        mean_intensity_2 = 50  # Dummy Value

        # Cell Classification (Example: Based on area and circularity)
        if (
            50 < area < 200
            and circularity > 0.6
            and 80 < mean_intensity_1 < 120
        ):
            cell_type_1_count += 1
            cv2.drawContours(
                counted_image, [contour], -1, (0, 255, 0), 2
            )  # Green for Type 1
        elif (
            100 < area < 500
            and circularity < 0.4
            and 40 < mean_intensity_2 < 60
        ):
            cell_type_2_count += 1
            cv2.drawContours(
                counted_image, [contour], -1, (255, 0, 0), 2
            )  # Blue for Type 2

    # Display the counted image.
    cv2.imshow("Cell Counts", counted_image)
    cv2.waitKey(0)  # Waits for key press
    cv2.destroyAllWindows()

    return {"Cell Type 1": cell_type_1_count, "Cell Type 2": cell_type_2_count}


def main(image_path):
    """
    Main function to load, process, and count cells in an image.

    Args:
        image_path (str): Path to the image file.
    """
    image = load_image(image_path)
    if image is None:
        print("Error: Image loading failed. Exiting.")
        return

    image_with_contours, contours = apply_contours(image)
    if contours is None:
        print("Error: Contour detection failed. Exiting.")
        return

    counts = perform_counts(image_with_contours, contours)
    if counts is None:
        print("Error: Cell counting failed. Exiting")
        return

    print(f"Cell Counts: {counts}")


if __name__ == "__main__":
    image_path = os.path.abspath(
        ""
    )  # create input function here to input the image.
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    else:
        main(image_path)
