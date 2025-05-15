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


def save_image_with_contours(image, show=False):
    r, _, b = cv.split(image)

    blurred_r = cv.GaussianBlur(r, (5, 5), 0)
    blurred_b = cv.GaussianBlur(b, (1, 1), 0)

    if blurred_r.dtype != np.uint8:
        blurred_r = (blurred_r / np.max(blurred_r) * 255).astype(np.uint8)
    if blurred_b.dtype != np.uint8:
        blurred_b = (blurred_b / np.max(blurred_b) * 255).astype(np.uint8)

    blue_edges = cv.Canny(blurred_b, 100, 200)
    contours_b, _ = cv.findContours(
        blue_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    red_edges = cv.Canny(blurred_r, 100, 200)
    contours_r, _ = cv.findContours(
        red_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    out = image.copy()
    count = [0, 0]  # [valid, total]
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

        count[1] += 1

        if is_inside:
            cv.drawContours(out, [cb], -1, (0, 255, 0), 1)
            count[0] += 1
        else:
            cv.drawContours(out, [cb], -1, (0, 0, 255), 1)  # mark invalid ones

    if show:
        cv.imshow("Contours", cv.cvtColor(out, cv.COLOR_BGR2RGB))
        print("Valid cells:", count)

        while True:
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cv.destroyAllWindows()

    return count


def main(image_path, show=False):
    """
    Main function to load and process the image.

    Args:
        image_path (str): Path to the image file.
    """
    image = load_image(image_path)
    if image is not None:
        count = save_image_with_contours(image, show)
        return count
    else:
        print(f"Error: Could not load image at {image_path}")
        sys.exit(1)


def menu():
    print("Select the type of processing:")
    print("1. Single image")
    print("2. Bulk")
    return input("Enter choice (1/2): ").strip()


def bulk_menu():
    print("Select bulk processing type:")
    print("1. Multiple slides")
    print("2. Multiple slides + multiple regions")
    return input("Enter choice (1/2): ").strip()


def process_single_image():
    path = input("Enter path to .vsi image: ").strip()
    if not os.path.exists(path) or not path.endswith("m.vsi"):
        print("Invalid path.")
        return
    count = main(path, True)
    print(f"Count: {count}")


def process_bulk_slides():
    base_dir = input(
        "Enter path to slide directory (e.g., ./data/Test4): "
    ).strip()
    out_path = input(
        "Enter full output CSV path (e.g., ./results/slides.csv): "
    ).strip()
    with open(out_path, "w") as file:
        writer = csv.writer(file)
        slides = os.listdir(base_dir)
        for slide in sorted(slides):
            image_path = os.path.join(base_dir, slide)
            if not image_path.endswith("m.vsi"):
                print(f"Skipping {slide}")
                continue
            if not os.path.exists(image_path):
                print(f"Not found: {image_path}")
                continue
            count = main(image_path)
            writer.writerow([slide, count])
    print(f"Done. Output saved to {out_path}")


def process_bulk_slides_regions():
    base_dir = input("Enter base data dir (e.g., ./data): ").strip()
    out_path = input(
        "Enter full output CSV path (e.g., ./results/full.csv): "
    ).strip()
    with open(out_path, "w") as file:
        writer = csv.writer(file)
        cols1 = os.listdir(base_dir)
        for col in sorted(cols1):
            slides = os.listdir(os.path.join(base_dir, col))
            for slide in sorted(slides):
                regions = os.listdir(os.path.join(base_dir, col, slide))
                for region in sorted(regions):
                    image_path = os.path.join(base_dir, col, slide, region)
                    if not image_path.endswith("m.vsi"):
                        print(f"Skipping {region}")
                        continue
                    if not os.path.exists(image_path):
                        print(f"Not found: {image_path}")
                        continue
                    count = main(image_path)
                    writer.writerow([col, slide, region, count])
    print(f"Done. Output saved to {out_path}")


if __name__ == "__main__":
    choice = menu()

    if choice == "1":
        process_single_image()
    elif choice == "2":
        sub_choice = bulk_menu()
        if sub_choice == "1":
            process_bulk_slides()
        elif sub_choice == "2":
            process_bulk_slides_regions()
        else:
            print("Invalid sub-choice.")
    else:
        print("Invalid choice.")

    javabridge.kill_vm()
