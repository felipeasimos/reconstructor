import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Generates a (size x size) Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolves a 2D image with a kernel."""
    image_h, image_w = image.shape
    kernel_w, kernel_h = kernel.shape
    pad_w, pad_h = kernel_h // 2, kernel_w // 2

    padded_image = np.pad(
        image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    result = np.zeros_like(image)

    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            result[i, j] = np.sum(region * kernel)

    return result


def apply_gaussian_blur(img: np.ndarray, kernel_size=5, sigma=1.0) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    if img.ndim == 2:  # Grayscale
        return convolve2d(img, kernel)
    elif img.ndim == 3:  # RGB
        return np.stack([convolve2d(img[:, :, c], kernel) for c in range(img.shape[2])], axis=2)


def grayscale(img):
    if img.ndim == 3:
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img.copy()


def display_image(image_path):
    """
    Reads and displays a PNG image using matplotlib.

    Parameters:
    image_path (str): Path to the PNG image file.
    """

    try:
        edge_detection_kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        # Read the image
        img = mpimg.imread(image_path)

        height, width, channels = img.shape
        print(f"Image dimensions: {width}x{height}, Channels: {channels}")
        gray = grayscale(img)

        kerne_size = 9
        sigma = 2.0
        blurred = apply_gaussian_blur(
            gray, kernel_size=kerne_size, sigma=sigma)
        edges = convolve2d(blurred, edge_detection_kernel)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 5))

        ax1.imshow(img[..., :3])
        ax1.set_title("Original", pad=10)
        ax1.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax2.imshow(gray, cmap="gray")
        ax2.set_title("Grayscale", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.7, color='grey')

        ax3.imshow(blurred, cmap="gray")
        ax3.set_title(
            f"After gaussian blur (kernel_size={kerne_size},sigma={sigma})")
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
