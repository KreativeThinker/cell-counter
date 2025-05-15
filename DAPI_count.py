import csv
import os
import sys

import bioformats
import cv2 as cv
import javabridge
import numpy as np

javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)


def load_image(path):
    """
    Load an image using Bio-Formats and return the image data.

    Parameters:
        path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image data.
    """
    # Start the Java Virtual Machine (JVM)

    # Read the image using Bio-Formats
    image_data = bioformats.load_image(path)

    return image_data


def save_image_with_contours(image):
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
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    out = image.copy()
    count = 0
    for c in contours:
        area = cv.contourArea(c)
        if area > 5:
            cv.drawContours(out, [c], -1, (0, 255, 0), 1)
            count += 1

    # print("Cells:", count)

    image_with_contours = image.copy()
    cv.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)

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
    # plt.imsave(f"{path[:-4]}.png", image_with_contours)

    # display = cv.cvtColor(cv.resize(out, (800, 600)), cv.COLOR_BGR2RGB)
    # cv.imshow("Contours", display)
    # while True:
    #     if cv.waitKey(1) & 0xFF == ord("q"):
    #         break
    cv.destroyAllWindows()
    # print("Image with contours saved as contours.png")
    return count


def main(image_path):
    """
    Main function to load and process the image.

    Args:
        image_path (str): Path to the image file.
    """
    image = load_image(image_path)
    if image is not None:
        count = save_image_with_contours(image)
        return count
    else:
        print(f"Error: Could not load image at {image_path}")
        sys.exit(1)


if __name__ == "__main__":
    # image_path = "./data/<filename>.vsi"
    # if not os.path.exists(image_path):
    #     print(f"Error: Image file not found at {image_path}")
    #     sys.exit(1)
    # main(image_path)

    with open("./ihc.csv", "w") as file:
        cols1 = os.listdir("./data")
        for col in sorted(cols1):
            slides = os.listdir(f"./data/{col}")
            for slide in sorted(slides):
                regions = os.listdir(f"./data/{col}/{slide}")
                for region in sorted(regions):
                    image_path = f"./data/{col}/{slide}/{region}"
                    if not image_path.endswith("d.vsi"):
                        print(f"file path not valid. Skipping {region}")
                        continue
                    if not os.path.exists(image_path):
                        print(f"Error: Image file not found at {image_path}")
                        sys.exit(1)
                    count = main(image_path)
                    writer = csv.writer(file)
                    writer.writerow([col, slide, region, count])

    javabridge.kill_vm()
