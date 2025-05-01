import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from grayscale import grayscale
from gaussian import gaussian_kernel
from common import convolve2d
import cv2


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

        kernel_size = 3
        sigma = 1.4
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 5))

        ax1.imshow(img)
        ax1.set_title("Original", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax2.imshow(gray, cmap="gray")
        ax2.set_title("Grayscale", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax3.imshow(blurred, cmap="gray")
        ax3.set_title(
            f"After gaussian blur (kernel_size={kernel_size},sigma={sigma})")
        ax3.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax4.imshow(edges, cmap="gray")
        ax4.set_title("After edge detection kernel")
        ax4.grid(True, linestyle="--", alpha=0.7, color='grey')

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
