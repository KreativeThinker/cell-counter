import logging
import os
import sys

import bioformats
import cv2 as cv
import javabridge
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    """
    Load an image using Bio-Formats and return the image data.

    Parameters:
        path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image data.
    """
    # Start the Java Virtual Machine (JVM)
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)

    try:
        # Read the image using Bio-Formats
        image_data = bioformats.load_image(path)
        omexml_metadata = bioformats.OMEXML()
        logging.info(omexml_metadata)

        return image_data
    finally:
        # Stop the JVM
        javabridge.kill_vm()


def save_image_with_contours(image, path):
    """
    Display the image with highlighted contours using matplotlib.

    Parameters:
        image (numpy.ndarray): The image data to display.
        path (str): The path to the image file.
    """
    # Convert the image to grayscale for contour detection
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # Use COLOR_RGB2GRAY
    # Apply a blur to reduce noise
    blurred = cv.GaussianBlur(gray, (1, 1), 0)
    # Ensure the image is of type CV_8U
    if blurred.dtype != np.uint8:
        blurred = (blurred / np.max(blurred) * 255).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv.Canny(blurred, 100, 200)  # Adjust thresholds as needed

    # Find contours
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    image_with_contours = image.copy()
    cv.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)

    print(f"Cell count: {len(contours)}")

    # Draw contours on a copy of the original image
    image_with_contours = image.copy()
    cv.drawContours(
        image_with_contours, contours, -1, (0, 255, 0), 1
    )  # Green contours

    # Ensure the image data is in the 0-1 range if it's float
    if (
        image_with_contours.dtype == np.float32
        or image_with_contours.dtype == np.float64
    ):
        image_with_contours = np.clip(image_with_contours, 0, 1)

    # Save the image with contours
    plt.imsave(f"{path[:-4]}.png", image_with_contours)
    print("Image with contours saved as contours.png")


def main(image_path):
    """
    Main function to load and process the image.

    Args:
        image_path (str): Path to the image file.
    """
    image = load_image(image_path)
    if image is not None:
        save_image_with_contours(image, image_path)
    else:
        print(f"Error: Could not load image at {image_path}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image_path = "./data/Ananya Sehgal_T1C1/Control 1/Slide 1/Au1_L2_C1m.vsi"
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    main(image_path)
    # overlay mask on original file
