import numpy as np


def gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Generates a (size x size) Gaussian kernel."""
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)
