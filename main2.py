import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ransac import ransac_line_clusters
import cv2
import itertools



def segment_to_line(line):
    """Convert line segment to line equation coefficients Ax + By = C."""
    x1, y1, x2, y2 = line[0]
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, C


def direction_vector(segment):
    """Get direction vector of a line segment."""
    x1, y1, x2, y2 = segment[0]
    return x2 - x1, y2 - y1


def get_intersection_point(line1, line2):
    """
    Find intersection point between two line segments.
    
    Returns None if lines are nearly parallel (sin(theta) < 0.1).
    """
    A1, B1, C1 = segment_to_line(line1)
    A2, B2, C2 = segment_to_line(line2)
    
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
    if abs(determinant) < 1e-10:
        return None
    
    x = (C1 * B2 - C2 * B1) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return x, y


def get_intersections(lines):
    """Generate all intersection points between pairs of lines."""
    for line1, line2 in itertools.combinations(lines, 2):
        point = get_intersection_point(line1, line2)
        if point is not None:
            yield point

def detect_lines(img):
    """Detect lines using Canny + Hough transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.2)
    edges = cv2.Canny(blurred, 50, 100)
    lines = cv2.HoughLinesP(edges, 1.5, np.pi/360, threshold=100, minLineLength=10, maxLineGap=10)
    return lines

def compute_vanishing_line(vp1, vp2):
    """Return vanishing line from two vanishing points (cross product in homogeneous coords)."""
    v1 = np.array([*vp1, 1.0])
    v2 = np.array([*vp2, 1.0])
    return np.cross(v1, v2)

def plane_from_vanishing_line(K, vline):
    """Get plane normal from vanishing line using K^T * l."""
    invK_T = np.linalg.inv(K).T
    normal = invK_T @ vline
    return normal / np.linalg.norm(normal)

def intersect_ray_plane(ray_origin, ray_dir, plane_normal, plane_point):
    """Intersect a ray with a plane (returns 3D point)."""
    d = np.dot(plane_normal, plane_point)
    t = (d - np.dot(plane_normal, ray_origin)) / np.dot(plane_normal, ray_dir)
    return ray_origin + t * ray_dir

def estimate_intrinsics(vx, vy, vz, img_shape):
    """Estimate focal length from orthogonal vanishing points."""
    cx, cy = img_shape[0] / 2, img_shape[1] / 2
    def dot(u, v): return np.dot(u - [cx, cy], v - [cx, cy])
    fx_squared = -dot(vx, vy)
    f = np.sqrt(abs(fx_squared))
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def backproject(K, pt_2d):
    """Backproject 2D point to 3D ray."""
    pt_h = np.array([*pt_2d, 1.0])
    ray = np.linalg.inv(K) @ pt_h
    return ray / np.linalg.norm(ray)

def plot_3d_scene(points_3d, texture_img, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        verts = [points_3d[i] for i in face]
        poly = Poly3DCollection([verts], facecolors='lightgray', edgecolor='k', alpha=0.8)
        ax.add_collection3d(poly)

    pts = np.array(points_3d)
    ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.view_init(elev=30, azim=45)
    plt.show()

def generate_plane_corners(clusters):
    """
    Generate corner candidates from intersections of lines from different vanishing direction clusters.
    Returns list of 2D points.
    """
    plane_corners = []

    for i, j in itertools.combinations(range(len(clusters)), 2):
        lines_i = clusters[i]
        lines_j = clusters[j]
        for li in lines_i:
            for lj in lines_j:
                p = get_intersection_point(li, lj)
                if p is not None:
                    plane_corners.append(p)

    return np.array(plane_corners)

def main():
    # 1. detect lines
    image = cv2.imread("examples/rubik.jpeg")
    lines = detect_lines(image)

    # 2. detect vanishing points
    clusters = ransac_line_clusters(lines, np.deg2rad(10), 3)

    intersection_points = []
    vanishing_points = []

    for cluster in clusters:
        intersections = list(get_intersections(cluster))
        intersection_points.extend(intersections)
        
        if intersections:
            vp = np.mean(intersections, axis=0)
            vanishing_points.append(vp)

    vp_x, vp_y, vp_z = vanishing_points
    # 3. calculate calibration matrix
    K = estimate_intrinsics(vp_x, vp_y, vp_z, image.shape)

    plane_corners = generate_plane_corners(clusters)

    vline_plane = compute_vanishing_line(vp_x, vp_z)
    plane_normal = plane_from_vanishing_line(K, vline_plane)
    corners_2d = plane_corners[:12]

    ref_2d = corners_2d[0]
    ref_ray = backproject(K, ref_2d)
    ref_3d = ref_ray * 2
    plane_point = ref_3d

    points_3d = []
    for pt in corners_2d:
        ray_dir = backproject(K, pt)
        pt3d = intersect_ray_plane(np.zeros(3), ray_dir, plane_normal, plane_point)
        points_3d.append(pt3d)

    faces = [
        list(range(i, i+4))
        for i in range(0, len(points_3d), 4)
        if i + 3 < len(points_3d)
    ]

    plot_3d_scene(points_3d, image, faces)

if __name__ == "__main__":
    main()
