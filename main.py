import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import numpy as np


def grayscale(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) if img.ndim == 3 else img.copy()


def convolution(img, kernel):
    pad_size = kernel.shape[0] // 2
    padded = np.pad(img, pad_size, mode='reflect')

    # Initialize output and apply convolution
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+3, j:j+3]  # 3x3 window for 3x3 kernel
            output[i, j] = np.sum(region * kernel)
    return np.clip(output, 0, 1)


def display_image(image_path):
    """
    Reads and displays a PNG image using matplotlib.

    Parameters:
    image_path (str): Path to the PNG image file.
    """

    try:
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        # Read the image
        img = mpimg.imread(image_path)

        height, width, channels = img.shape
        print(f"Image dimensions: {width}x{height}, Channels: {channels}")
        gray = grayscale(img)
        output = convolution(gray, kernel)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(img[..., :3])
        ax1.set_title("Original", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax2.imshow(gray, cmap="gray")
        ax2.set_title("Grayscale", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax3.imshow(output, cmap="gray")
        ax3.set_title("After edge detection kernel", pad=10)
        ax3.grid(True, linestyle="--", alpha=0.7, color='grey')

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
