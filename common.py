import numpy as np


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
