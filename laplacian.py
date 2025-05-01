import numpy as np
from common import convolve2d


def laplacian_edge_detection(img: np.ndarray) -> np.ndarray:
    edge_detection_kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    return convolve2d(img, edge_detection_kernel)
