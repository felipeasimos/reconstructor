import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import itertools


def is_grayscale(img):
    return len(img.shape) == 2


def get_image_in_rgb(img):
    if is_grayscale(img):
        return cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    else:
        return img.copy()


def get_image_with_lines(img, lines):
    # Draw the lines on a copy of the original image
    output = get_image_in_rgb(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return output


def segment_to_line(line):
    x1, y1, x2, y2 = line[0]
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, C


def direction_vector(segment):
    x1, y1, x2, y2 = segment[0]
    return x2 - x1, y2 - y1


def get_intersection_point(line1, line2):
    A1, B1, C1 = segment_to_line(line1)
    A2, B2, C2 = segment_to_line(line2)
    # v1 x v2 = ||v1|| ||v2|| sin(theta)
    v1 = direction_vector(line1)
    v2 = direction_vector(line2)
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    len1 = np.hypot(*v1)
    len2 = np.hypot(*v2)
    sin_theta = cross / (len1 * len2)
    if (abs(sin_theta) < 0.1):
        return None
    determinant = A1 * B2 - A2 * B1
    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return x, y


def get_intersections(lines):
    for (line1, line2) in itertools.combinations(lines, 2):
        point = get_intersection_point(line1, line2)
        if (point is not None):
            yield point


def display_image(image_path):
    """
    Reads and displays a PNG image using matplotlib.

    Parameters:
    image_path (str): Path to the PNG image file.
    """

    try:
        # Read the image
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        height, width, channels = img.shape
        print(f"Image dimensions: {width}x{height}, Channels: {channels}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel_size = 9
        sigma = 1.4
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        edges = cv2.Canny(blurred, threshold1=100,
                          threshold2=200)
        lines = cv2.HoughLinesP(
            edges, rho=3, theta=np.pi / 180, threshold=150, minLineLength=100, maxLineGap=10)

        hough_edges = get_image_with_lines(edges, lines)
        intersection_points = list(get_intersections(lines))
        print(f"{len(intersection_points)} intersection points found")

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)
              ) = plt.subplots(3, 2, figsize=(15, 5))

        ax1.imshow(img)
        ax1.set_title("Original")
        ax1.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax2.imshow(gray, cmap="gray")
        ax2.set_title("Grayscale")
        ax2.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax3.imshow(blurred, cmap="gray")
        ax3.set_title(
            f"After gaussian blur (kernel_size={kernel_size},sigma={sigma})")
        ax3.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax4.imshow(edges, cmap="gray")
        ax4.set_title("After edge detection kernel")
        ax4.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax5.imshow(hough_edges, cmap="gray")
        ax5.set_title("hough lines")
        ax5.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax6.imshow(hough_edges, cmap="gray")
        for point in intersection_points:
            x, y = point
            ax6.scatter(x, y, c='red', s=10)
        height, width = hough_edges.shape[:2]
        ax6.set_xlim(0, width)
        ax6.set_ylim(height, 0)
        ax6.set_title("vanishing points")
        ax6.grid(True, linestyle="--", alpha=0.7, color='grey')

        plt.tight_layout()
        plt.show()

        # # Display the image
        # plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
        # plt.imshow(img)
        # plt.axis('on')  # Turn off axis labels
        # plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Display a PNG image using matplotlib.")

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the PNG image file to display"
    )

    args = parser.parse_args()
    display_image(args.image_path)


main()
