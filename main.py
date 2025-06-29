"""
Image line detection and vanishing point analysis using Hough transform and RANSAC clustering.

This module provides functionality to detect lines in images, find their intersections,
and identify vanishing points using computer vision techniques.
"""

import argparse
import itertools
from typing import List, Optional, Tuple, Generator

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ransac import ransac_line_clusters
from image_viewer import ImageViewer


def is_grayscale(img: np.ndarray) -> bool:
    """Check if an image is grayscale."""
    return len(img.shape) == 2


def get_image_in_rgb(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB format."""
    if is_grayscale(img):
        return cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    return img.copy()


def get_image_with_lines(img: np.ndarray, lines: Optional[np.ndarray]) -> np.ndarray:
    """Draw detected lines on a copy of the original image."""
    output = get_image_in_rgb(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return output


def segment_to_line(line: np.ndarray) -> Tuple[float, float, float]:
    """Convert line segment to line equation coefficients Ax + By = C."""
    x1, y1, x2, y2 = line[0]
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, C


def direction_vector(segment: np.ndarray) -> Tuple[float, float]:
    """Get direction vector of a line segment."""
    x1, y1, x2, y2 = segment[0]
    return x2 - x1, y2 - y1


def get_intersection_point(line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find intersection point between two line segments.
    
    Returns None if lines are nearly parallel (sin(theta) < 0.1).
    """
    A1, B1, C1 = segment_to_line(line1)
    A2, B2, C2 = segment_to_line(line2)
    
    # Check if lines are nearly parallel using cross product
    v1 = direction_vector(line1)
    v2 = direction_vector(line2)
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    len1 = np.hypot(*v1)
    len2 = np.hypot(*v2)
    
    if len1 == 0 or len2 == 0:
        return None
    
    sin_theta = cross / (len1 * len2)
    if abs(sin_theta) < 0.1:
        return None
    
    determinant = A1 * B2 - A2 * B1
    if abs(determinant) < 1e-10:  # Lines are parallel
        return None
    
    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return x, y


def get_intersections(lines: List[np.ndarray]) -> Generator[Tuple[float, float], None, None]:
    """Generate all intersection points between pairs of lines."""
    for line1, line2 in itertools.combinations(lines, 2):
        point = get_intersection_point(line1, line2)
        if point is not None:
            yield point


def process_image(image_path: str) -> None:
    """
    Process an image to detect lines and find vanishing points.
    
    Args:
        image_path: Path to the image file
    """
    # Image processing parameters
    KERNEL_SIZE = 9
    SIGMA = 1.2
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 100
    HOUGH_RHO = 1.5
    HOUGH_THETA = np.pi / 360
    HOUGH_THRESHOLD = 100
    MIN_LINE_LENGTH = 10
    MAX_LINE_GAP = 10
    ANGLE_THRESHOLD = np.deg2rad(10)
    MAX_CLUSTERS = 3
    
    try:
        # Read and convert image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        print(f"Image dimensions: {width}x{height}, Channels: {channels}")
        
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (KERNEL_SIZE, KERNEL_SIZE), SIGMA)
        
        # Edge detection and line detection
        edges = cv2.Canny(blurred, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        lines = cv2.HoughLinesP(
            edges, 
            rho=HOUGH_RHO, 
            theta=HOUGH_THETA, 
            threshold=HOUGH_THRESHOLD,
            minLineLength=MIN_LINE_LENGTH, 
            maxLineGap=MAX_LINE_GAP
        )
        
        if lines is None:
            print("No lines detected in the image")
            return
        
        hough_edges = get_image_with_lines(edges, lines)
        
        # Cluster lines and find vanishing points
        intersection_points = []
        vanishing_points = []
        
        clusters = ransac_line_clusters(lines, ANGLE_THRESHOLD, MAX_CLUSTERS)
        
        for cluster in clusters:
            intersections = list(get_intersections(cluster))
            intersection_points.extend(intersections)
            
            if intersections:
                vp = np.mean(intersections, axis=0)
                vanishing_points.append(vp)
        
        print(f"{len(intersection_points)} intersection points found")
        print(f"{len(vanishing_points)} vanishing points found")
        
        # Display results with interactive viewer
        _display_results_interactive(img, gray, blurred, edges, hough_edges, intersection_points, vanishing_points)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def _display_results_interactive(
    original_img: np.ndarray,
    gray_img: np.ndarray,
    blurred_img: np.ndarray,
    edges: np.ndarray,
    hough_edges: np.ndarray,
    intersection_points: List[Tuple[float, float]],
    vanishing_points: List[Tuple[float, float]]
) -> None:
    """Display the processing steps using an interactive viewer."""
    
    def manual_line_callback(manual_lines, intersections, vanishing_point):
        """Callback function to handle manually drawn lines and calculated vanishing points."""
        print(f"\n📋 Manual Line Analysis Results:")
        print(f"   Lines drawn: {len(manual_lines)}")
        print(f"   Intersections found: {len(intersections)}")
        print(f"   Vanishing point: ({vanishing_point[0]:.2f}, {vanishing_point[1]:.2f})")
        print(f"   Compare with Hough-based vanishing points: {len(vanishing_points)} found")
    
    # Prepare images for the viewer
    images = [
        {
            'data': original_img,
            'title': 'Original Image - Draw Lines Here (M to toggle)',
            'show_grid': False,
            'show_axis': False,
        },
        {
            'data': gray_img,
            'title': 'Grayscale Image',
            'cmap': 'gray',
            'show_grid': True,
            'show_axis': True,
        },
        {
            'data': blurred_img,
            'title': 'Gaussian Blurred',
            'cmap': 'gray',
            'show_grid': True,
            'show_axis': True,
        },
        {
            'data': edges,
            'title': 'Canny Edge Detection',
            'cmap': 'gray',
            'show_grid': True,
            'show_axis': True,
        },
        {
            'data': hough_edges,
            'title': 'Hough-Detected Lines with Points',
            'cmap': 'gray',
            'intersection_points': intersection_points,
            'vanishing_points': vanishing_points,
            'show_grid': True,
            'show_axis': True,
        },
        {
            'data': None,
            'title': '3D Scene Reconstruction - Initial View',
            'is_3d': True,
            'show_grid': True,
            'show_axis': True,
        },
        {
            'data': None,
            'title': '3D Scene Reconstruction - Final View',
            'is_3d': True,
            'show_grid': True,
            'show_axis': True,
        }
    ]
    
    # Create and show the interactive viewer
    print("\n🖼️  Interactive Image Processing Pipeline Viewer")
    print("📖 Navigation: Use arrow keys (← →) or A/D to navigate between steps")
    print("🖊️  Manual Drawing: Press 'M' to toggle drawing mode, 'C' to clear, 'V' to calculate VP")
    print("❌ Press 'q' or Escape to quit")
    
    viewer = ImageViewer(images, line_callback=manual_line_callback)
    viewer.show()


def main() -> None:
    """Main function to parse arguments and process the image."""
    parser = argparse.ArgumentParser(
        description="Detect lines and vanishing points in an image using computer vision techniques."
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to process"
    )
    
    args = parser.parse_args()
    process_image(args.image_path)


if __name__ == "__main__":
    main()
