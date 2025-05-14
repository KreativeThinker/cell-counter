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
    r, _, b = cv.split(image)

    blurred_r = cv.GaussianBlur(r, (5, 5), 0)
    blurred_b = cv.GaussianBlur(b, (1, 1), 0)

    if blurred_r.dtype != np.uint8:
        blurred_r = (blurred_r / np.max(blurred_r) * 255).astype(np.uint8)
    if blurred_b.dtype != np.uint8:
        blurred_b = (blurred_b / np.max(blurred_b) * 255).astype(np.uint8)

    # cv.imshow("bluelayer", blurred_b)
    # cv.imshow("redlayer", blurred_r)

    blue_edges = cv.Canny(blurred_b, 100, 200)
    contours_b, _ = cv.findContours(
        blue_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    red_edges = cv.Canny(blurred_r, 100, 200)
    contours_r, _ = cv.findContours(
        red_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    out = image.copy()
    count = 0
    for cb in contours_b:
        area = cv.contourArea(cb)
        if area <= 5:
            continue

        # Get centroid of blue contour
        M = cv.moments(cb)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        point = (cx, cy)

        # Check if point is inside or on any red contour
        is_inside = any(
            cv.pointPolygonTest(cr, point, False) >= 0 for cr in contours_r
        )

        cv.drawContours(out, contours_r, -1, (255, 0, 0), 1)

        if is_inside:
            cv.drawContours(out, [cb], -1, (0, 255, 0), 1)
            count += 1
        else:
            cv.drawContours(out, [cb], -1, (0, 0, 255), 1)  # mark invalid ones

    # cv.imshow("Contours", cv.cvtColor(out, cv.COLOR_BGR2RGB))
    # print("Valid cells:", count)
    #
    # while True:
    #     if cv.waitKey(1) & 0xFF == ord("q"):
    #         break
    # cv.destroyAllWindows()

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
    # image_path = "./data/Test 4/Slide 2/Au1_L3_T3m.vsi"
    # if not os.path.exists(image_path):
    #     print(f"Error: Image file not found at {image_path}")
    #     sys.exit(1)
    # main(image_path)

    with open("./tritC_5x5.csv", "w") as file:
        cols1 = os.listdir("./data")
        for col in sorted(cols1):
            slides = os.listdir(f"./data/{col}")
            for slide in sorted(slides):
                regions = os.listdir(f"./data/{col}/{slide}")
                for region in sorted(regions):
                    image_path = f"./data/{col}/{slide}/{region}"
                    if not image_path.endswith("m.vsi"):
                        print(f"file path not valid. Skipping {region}")
                        continue
                    if not os.path.exists(image_path):
                        print(f"Error: Image file not found at {image_path}")
                        sys.exit(1)
                    count = main(image_path)
                    writer = csv.writer(file)
                    writer.writerow([col, slide, region, count])

    javabridge.kill_vm()
