"""
Vanishing point calculation utilities for line intersection and clustering analysis.

This module provides functions to calculate intersections between lines and find
vanishing points using RANSAC clustering techniques.
"""

import itertools
from typing import List, Optional, Tuple, Generator, Union

import numpy as np

from ransac import ransac_line_clusters


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


def calculate_vanishing_points_from_lines(
    lines: List[Union[np.ndarray, List[Tuple[float, float]]]],
    angle_threshold: float = np.deg2rad(10),
    max_clusters: int = 3
) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]:
    """
    Calculate vanishing points from a list of lines using RANSAC clustering.
    
    Args:
        lines: List of line segments. Each line can be either:
               - np.ndarray with shape [[x1, y1, x2, y2]] (Hough format)
               - List of two tuples [(x1, y1), (x2, y2)] (manual drawing format)
        angle_threshold: Maximum angle difference for lines to be in same cluster
        max_clusters: Maximum number of clusters to find
    
    Returns:
        Tuple of (vanishing_points, intersection_points_per_cluster)
        - vanishing_points: List of (x, y) coordinates for each vanishing point
        - intersection_points_per_cluster: List of intersection points for each cluster
    """
    # Convert manual drawing format to Hough format if needed
    normalized_lines = []
    for line in lines:
        if isinstance(line, np.ndarray):
            # Already in Hough format [[x1, y1, x2, y2]]
            normalized_lines.append(line)
        else:
            # Convert from manual drawing format [(x1, y1), (x2, y2)]
            start, end = line
            line_segment = np.array([[int(start[0]), int(start[1]), int(end[0]), int(end[1])]])
            normalized_lines.append(line_segment)
    
    # Use RANSAC to cluster lines by direction
    clusters = ransac_line_clusters(normalized_lines, angle_threshold, max_clusters)
    
    # Calculate vanishing points for each cluster
    vanishing_points = []
    intersection_points_per_cluster = []
    
    for cluster in clusters:
        # Calculate intersections within this cluster
        cluster_intersections = list(get_intersections(cluster))
        intersection_points_per_cluster.append(cluster_intersections)
        
        if cluster_intersections:
            # Calculate vanishing point as mean of intersections in this cluster
            vp = np.mean(cluster_intersections, axis=0)
            vanishing_points.append((vp[0], vp[1]))
    
    return vanishing_points, intersection_points_per_cluster


def calculate_vanishing_points_from_intersections(
    intersection_points: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Calculate a single vanishing point from a list of intersection points.
    
    This is the old method that just takes the mean of all intersections.
    Use calculate_vanishing_points_from_lines() for the RANSAC-based approach.
    
    Args:
        intersection_points: List of (x, y) intersection coordinates
    
    Returns:
        Single vanishing point as (x, y) coordinates
    """
    if not intersection_points:
        raise ValueError("No intersection points provided")
    
    vp = np.mean(intersection_points, axis=0)
    return vp[0], vp[1]

def compute_focal_length(v1, v2, v3):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)


    def get_constraint(vp1, vp2):
        x1, y1, _ = vp1
        x2, y2, _ = vp2
        return x1*x2 + y1*y2

    A = np.array([
        get_constraint(v1, v2),
        get_constraint(v1, v3),
        get_constraint(v2, v3)
    ])

    f_squared = -np.mean(A)
    f = np.sqrt(f_squared) if f_squared > 0 else None

    return f

