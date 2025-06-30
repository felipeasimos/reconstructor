"""
Interactive image viewer for displaying multiple images with keyboard navigation and manual line drawing.

This module provides an ImageViewer class that allows users to navigate through
a sequence of images using keyboard controls and draw lines manually on images.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools


class ImageViewer:
    """
    Interactive viewer for displaying multiple images with navigation controls and manual line drawing.
    
    Supports keyboard navigation:
    - Left/Right arrows or A/D: Navigate between images
    - Q/Escape: Quit the viewer
    - M: Toggle manual line drawing mode
    - C: Clear manually drawn lines
    - V: Calculate vanishing points from manual lines
    """
    
    def __init__(self, images: List[Dict[str, Any]], line_callback: Optional[Callable] = None):
        """
        Initialize the image viewer.
        
        Args:
            images: List of dictionaries containing image data and metadata.
                   Each dict should have 'data', 'title', and optionally 'cmap' keys.
            line_callback: Optional callback function to handle manual lines for vanishing point calculation.
        """
        self.images = images
        self.current_index = 0
        self.fig = None
        self.ax = None
        self.line_callback = line_callback
        
        # Manual line drawing state
        self.drawing_mode = False
        self.manual_lines = []
        self.current_line_points = []
        self.line_artists = []
        self.vanishing_point_artists = []  # Track VP markers
        self._manual_vp_coords = None  # Store VP coordinates for redrawing
        self.preview_line = None  # Track preview line while drawing
        
        # Mouse event handlers
        self.press_event = None
        self.motion_event = None
        self.release_event = None
        
    def _on_key_press(self, event):
        """Handle keyboard events for navigation and drawing mode."""
        if event.key in ['right', 'd']:
            self.current_index = (self.current_index + 1) % len(self.images)
            self._update_display()
        elif event.key in ['left', 'a']:
            self.current_index = (self.current_index - 1) % len(self.images)
            self._update_display()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)
        elif event.key == 'm':
            self._toggle_drawing_mode()
        elif event.key == 'c':
            self._clear_manual_lines()
        elif event.key == 'v':
            self._calculate_vanishing_points()
    
    def _toggle_drawing_mode(self):
        """Toggle manual line drawing mode."""
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self._connect_drawing_events()
            print("🖊️  Manual line drawing mode ON - Click and drag to draw lines")
        else:
            self._disconnect_drawing_events()
            print("🖊️  Manual line drawing mode OFF")
        
        # Update title to show drawing mode status
        self._update_display()
    
    def _connect_drawing_events(self):
        """Connect mouse events for line drawing."""
        self.press_event = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.motion_event = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.release_event = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
    
    def _disconnect_drawing_events(self):
        """Disconnect mouse events for line drawing."""
        # Clean up preview line when exiting drawing mode
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
            self.fig.canvas.draw()
        
        if self.press_event:
            self.fig.canvas.mpl_disconnect(self.press_event)
        if self.motion_event:
            self.fig.canvas.mpl_disconnect(self.motion_event)
        if self.release_event:
            self.fig.canvas.mpl_disconnect(self.release_event)
    
    def _on_press(self, event):
        """Handle mouse press for line drawing."""
        if not self.drawing_mode or event.inaxes != self.ax:
            return
        self.current_line_points = [(event.xdata, event.ydata)]
    
    def _on_motion(self, event):
        """Handle mouse motion for line drawing preview."""
        if not self.drawing_mode or event.inaxes != self.ax or not self.current_line_points:
            return
        
        # Remove previous preview line
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
        
        # Draw preview line from start point to current mouse position
        start_point = self.current_line_points[0]
        current_point = (event.xdata, event.ydata)
        
        self.preview_line, = self.ax.plot([start_point[0], current_point[0]], 
                                         [start_point[1], current_point[1]], 
                                         'r--', linewidth=3, alpha=0.5)
        self.fig.canvas.draw()
    
    def _on_release(self, event):
        """Handle mouse release to complete line drawing."""
        if not self.drawing_mode or event.inaxes != self.ax or not self.current_line_points:
            return
        
        # Remove preview line
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
        
        # Complete the line
        end_point = (event.xdata, event.ydata)
        start_point = self.current_line_points[0]
        
        # Only add line if it's not too short
        distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        if distance > 10:  # Minimum line length
            line_data = [start_point, end_point]
            self.manual_lines.append(line_data)
            
            # Draw the final line
            line_artist, = self.ax.plot([start_point[0], end_point[0]], 
                                      [start_point[1], end_point[1]], 
                                      'r-', linewidth=2, alpha=0.8)
            self.line_artists.append(line_artist)
            
            print(f"✏️  Line drawn: ({start_point[0]:.1f}, {start_point[1]:.1f}) to ({end_point[0]:.1f}, {end_point[1]:.1f})")
            self.fig.canvas.draw()
        
        self.current_line_points = []
    
    def _clear_manual_lines(self):
        """Clear all manually drawn lines and vanishing points."""
        self.manual_lines = []
        
        # Remove preview line if active
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
        
        # Remove line artists from plot
        for artist in self.line_artists:
            artist.remove()
        self.line_artists = []
        
        # Remove vanishing point artists from plot
        for artist in self.vanishing_point_artists:
            artist.remove()
        self.vanishing_point_artists = []
        
        # Clear stored VP coordinates
        self._manual_vp_coords = None
        
        print("🗑️  All manual lines and vanishing points cleared")
        self.fig.canvas.draw()
    
    def _calculate_vanishing_points(self):
        """Calculate vanishing points from manually drawn lines."""
        if len(self.manual_lines) < 2:
            print("⚠️  Need at least 2 lines to calculate vanishing points")
            return
        
        # Convert manual lines to format compatible with intersection calculation
        lines_for_calculation = []
        for line_points in self.manual_lines:
            start, end = line_points
            # Format: [[x1, y1, x2, y2]]
            line_segment = [[int(start[0]), int(start[1]), int(end[0]), int(end[1])]]
            lines_for_calculation.append(line_segment)
        
        # Calculate intersections
        intersections = list(self._get_line_intersections(lines_for_calculation))
        
        if intersections:
            # Calculate vanishing point as mean of intersections
            vp = np.mean(intersections, axis=0)
            
            # Store VP coordinates for redrawing
            self._manual_vp_coords = vp
            
            # Clear existing vanishing point artists and redraw
            for artist in self.vanishing_point_artists:
                artist.remove()
            self.vanishing_point_artists = []
            
            # Draw new vanishing point
            vp_artist = self.ax.scatter(vp[0], vp[1], c='red', s=50, marker='x', 
                                      linewidth=3, label='Manual VP')
            self.vanishing_point_artists.append(vp_artist)
            
            # Update legend
            self.ax.legend()
            
            print(f"📍 Vanishing point calculated: ({vp[0]:.1f}, {vp[1]:.1f})")
            print(f"📊 Based on {len(intersections)} intersection points from {len(self.manual_lines)} lines")
            
            # Call callback if provided
            if self.line_callback:
                self.line_callback(self.manual_lines, intersections, vp)
            
            self.fig.canvas.draw()
        else:
            print("❌ No valid intersections found between the drawn lines")
    
    def _get_line_intersections(self, lines):
        """Calculate intersections between pairs of lines."""
        for line1, line2 in itertools.combinations(lines, 2):
            intersection = self._get_line_intersection(line1, line2)
            if intersection is not None:
                yield intersection
    
    def _get_line_intersection(self, line1, line2):
        """Calculate intersection point between two line segments."""
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]
        
        # Convert to line equations: Ax + By = C
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1
        
        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3
        
        # Check if lines are parallel
        determinant = A1 * B2 - A2 * B1
        if abs(determinant) < 1e-10:
            return None
        
        # Calculate intersection
        x = (C1 * B2 - C2 * B1) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        
        return x, y

    def _draw_3d_grid(self):
        current_image = self.images[self.current_index]
        image_data = current_image['data']
        title = current_image['title']
        cmap = current_image.get('cmap', None)
        # Clear the figure and create new 3D axis
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Set up empty 3D plot with basic structure
        # Create a simple coordinate system for now
        x_range = np.linspace(-5, 5, 10)
        y_range = np.linspace(-5, 5, 10)
        z_range = np.linspace(0, 5, 10)

        # Create grid for visualization
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)  # Ground plane at z=0

        # Plot ground plane
        self.ax.plot_surface(X, Y, Z, alpha=0.3, color='lightgray')

        # Add coordinate axes
        self.ax.plot([0, 5], [0, 0], [0, 0], 'r-', linewidth=2, label='X-axis')
        self.ax.plot([0, 0], [0, 5], [0, 0], 'g-', linewidth=2, label='Y-axis')
        self.ax.plot([0, 0], [0, 0], [0, 5], 'b-', linewidth=2, label='Z-axis')

        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f"{title} ({self.current_index + 1}/{len(self.images)})")

        # Set view angle for better visualization
        self.ax.view_init(elev=20, azim=45)

        # Add legend
        self.ax.legend()

        # Show grid if specified
        show_grid = current_image.get('show_grid', True)
        if show_grid:
            self.ax.grid(True)

    def _estimate_focal_length(self, vps, center):
        vp1, vp2, vp3 = vps
        vp1_norm = vp1 - center
        vp2_norm = vp2 - center
        vp3_norm = vp3 - center
        
        equations = []
        vanishing_pairs = itertools.combinations(vps, 2)
        
        for vp_a, vp_b in vanishing_pairs:
            equations.append(vp_a[0]*vp_b[0] + vp_a[1]*vp_b[1])
        
        f_squared = -np.mean(equations)
        self.focal_length = np.sqrt(f_squared) if f_squared > 0 else max(self.image_width, self.image_height)
        return self.focal_length

    def _compute_vanishing_directions(self, K, vp1, vp2, vp3):
        vp1_h = np.array([vp1[0], vp1[1], 1])
        vp2_h = np.array([vp2[0], vp2[1], 1])
        vp3_h = np.array([vp3[0], vp3[1], 1])

        K_inv = np.linalg.inv(K)
        dir1 = K_inv @ vp1_h
        dir2 = K_inv @ vp2_h
        dir3 = K_inv @ vp3_h

        return dir1/np.linalg.norm(dir1), dir2/np.linalg.norm(dir2), dir3/np.linalg.norm(dir3)

    def _visualize_reconstruction(self, vertices, directions, vanishing_points):

        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(600, 0)
        colors = ['ro', 'go', 'bo']
        for i, vp in enumerate(vanishing_points):
            self.ax.plot(vp[0], vp[1], colors[i], markersize=10, label=f'VP{i+1}')
        self.ax.legend()
        self.ax.grid(True)
        
        # ax2 = fig.add_subplot(132, projection='3d')
        # ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=50)
        #
        # edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
        # for edge in edges:
        #     points = vertices[edge]
        #     ax2.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.6)
        #
        # origin = np.array([0, 0, 0])
        # colors = ['red', 'green', 'blue']
        # for i, direction in enumerate(directions):
        #     ax2.quiver(origin[0], origin[1], origin[2], 
        #                direction[0], direction[1], direction[2], 
        #                color=colors[i], arrow_length_ratio=0.1)
        # ax2.set_title('3D Reconstruction')
        #
        # ax3 = fig.add_subplot(133, projection='3d')
        # ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=50)
        # for edge in edges:
        #     points = vertices[edge]
        #     ax3.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.6)
        # ax3.view_init(elev=20, azim=45)
        # ax3.set_title('Different View')
        #
        # plt.tight_layout()
        # plt.show()
    def _enforce_orthogonality(self, directions):
        dir1, dir2, dir3 = directions
        # Start with first direction as-is
        u1 = dir1 / np.linalg.norm(dir1)
        
        # Orthogonalize second direction against first
        u2 = dir2 - np.dot(dir2, u1) * u1
        u2 = u2 / np.linalg.norm(u2)
        
        # Orthogonalize third direction against first two
        u3 = dir3 - np.dot(dir3, u1) * u1 - np.dot(dir3, u2) * u2
        u3 = u3 / np.linalg.norm(u3)
        
        return u1, u2, u3

    def _find_rectangle_corners(self, lines1, lines2, image_size):
        """Find corners formed by intersection of two sets of parallel lines"""
        w, h = image_size
        corners = []
        
        for rho1, theta1 in lines1:
            for rho2, theta2 in lines2:
                # Find intersection of two lines
                A = np.array([
                    [np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]
                ])
                b = np.array([rho1, rho2])
                
                try:
                    intersection = np.linalg.solve(A, b)
                    x, y = intersection
                    
                    # Check if intersection is within image bounds
                    if 0 <= x < w and 0 <= y < h:
                        corners.append([x, y])
                except np.linalg.LinAlgError:
                    continue
        
        # Sort corners to form a proper quadrilateral
        if len(corners) >= 4:
            corners = np.array(corners)
            # Sort by x-coordinate, then y-coordinate
            center = np.mean(corners, axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            return corners[sorted_indices][:4]
        
        return corners

    def _find_face_corners(self, line_groups, image_shape):
        """Find face corners from grouped lines"""
        h, w = image_shape[:2]
        
       
        corners = {}
        
        # Try to find rectangular regions formed by intersecting line groups
        if len(line_groups['x']) >= 2 and len(line_groups['y']) >= 2:
            # Find front face (formed by x and y direction lines)
            front_corners = self._find_rectangle_corners(
                line_groups['x'][:2], line_groups['y'][:2], (w, h)
            )
            if len(front_corners) == 4:
                corners['front'] = front_corners
        
        if len(line_groups['y']) >= 2 and len(line_groups['z']) >= 2:
            # Find right face (formed by y and z direction lines)  
            right_corners = self._find_rectangle_corners(
                line_groups['y'][:2], line_groups['z'][:2], (w, h)
            )
            if len(right_corners) == 4:
                corners['right'] = right_corners
                
        if len(line_groups['x']) >= 2 and len(line_groups['z']) >= 2:
            # Find top face (formed by x and z direction lines)
            top_corners = self._find_rectangle_corners(
                line_groups['x'][:2], line_groups['z'][:2], (w, h)
            )
            if len(top_corners) == 4:
                corners['top'] = top_corners
        
        return corners

    def line_to_rho_theta(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        # Direction vector of the line
        dx = x2 - x1
        dy = y2 - y1

        # Normal vector (perpendicular to direction)
        nx = -dy
        ny = dx

        # Normalize normal vector
        norm = np.hypot(nx, ny)
        nx /= norm
        ny /= norm

        # Midpoint of the line
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        # Rho = projection of midpoint onto the normal
        rho = mx * nx + my * ny

        # Theta = angle of the normal
        theta = np.arctan2(ny, nx)

        return rho, theta
    def _group_lines_by_vanishing_point(self, lines, directions, angle_threshold=10):
        """Group detected lines by which vanishing point they converge to"""
        keys = ['x', 'y', 'z']
        line_groups = {'x': [], 'y': [], 'z': []}
        for line in lines:
            rho, theta = self.line_to_rho_theta(line[0][0:2], line[0][2:])

            # Convert to line direction
            line_dir = np.array([-np.sin(theta), np.cos(theta)])

            # Find which vanishing point direction this line is closest to
            best_match = None
            best_similarity = -1

            for vp_name, vp_dir in enumerate(directions):

                # Project vp_dir to image plane for comparison
                vp_2d = vp_dir[:2] / (vp_dir[2] + 1e-8)
                vp_2d = vp_2d / np.linalg.norm(vp_2d)

                similarity = abs(np.dot(line_dir, vp_2d))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = vp_name

            if best_similarity > np.cos(np.radians(angle_threshold)):
                line_groups[keys[best_match]].append((rho, theta))

        return line_groups

    def _detect_face_corners(self, image, lines, directions):
        # Group lines by vanishing point
        line_groups = self._group_lines_by_vanishing_point(lines, directions)
        
        # Find intersections to get corners
        face_corners = self._find_face_corners(line_groups, image.shape)
            
        return face_corners

    def _visualize_detection(self, image, lines, face_corners):
        """Visualize line detection and corner detection results"""
        vis_image = image.copy()
        
        # Draw detected lines
        if lines is not None:
            for line in lines:
                rho, theta = self.line_to_rho_theta(line[0][0:2], line[0][2:])
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                self.ax.plot((x1, y1), (x2, y2), marker='o')
        
        # Draw detected corners
        colors = {'front': (0, 0, 1), 'right': (1, 0, 0), 'top': (0, 0, 1)}
        for face_name, corners in face_corners.items():
            color = colors.get(face_name, (255, 255, 255))
            corners = np.array(corners, dtype=np.int32)
            
            # Draw corners
            for corner in corners:
                self.ax.plot(corner[0], corner[1], 'o', color=color, markersize=5)
                pass
            
            # Draw face outline
            if len(corners) >= 4:
                polygon = list(corners) + [corners[0]]
                xs, ys = zip(*polygon)
                self.ax.plot(xs, ys, color=color, linewidth=2)
        
    def _update_3d_plot(self):
        self._draw_3d_grid()
        current_image = self.images[self.current_index]
        image_data = current_image['data']
        title = current_image['title']
        cmap = current_image.get('cmap', None)
        vps = current_image['vanishing_points']
        lines = current_image['lines']

        cx, cy, _ = image_data.shape
        center = np.array([cx/2, cy/2])
        focal_length = self._estimate_focal_length(vps, center)
        K = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])
        directions = self._compute_vanishing_directions(K, vps[0], vps[1], vps[2])
        # directions = self._enforce_orthogonality(directions)
        face_corners = self._detect_face_corners(image_data, lines, directions)
        self._visualize_detection(image_data, lines, face_corners)
        # Update the display
        self.fig.canvas.draw()

    def _update_display(self):
        """Update the displayed image."""
        current_image = self.images[self.current_index]
        image_data = current_image['data']
        title = current_image['title']
        cmap = current_image.get('cmap', None)
        is_3d = current_image.get('is_3d', False)
        
        # Handle 2D images (original logic)
        # Clear the figure and create new 2D axis
        self.fig.clear()

        # Handle 3D plots
        if is_3d:
            self._update_3d_plot()
            return

        self.ax = self.fig.add_subplot(111)

        # Display the image
        if cmap:
            self.ax.imshow(image_data, cmap=cmap)
        else:
            self.ax.imshow(image_data)
        
        # Handle overlay points if present
        if 'intersection_points' in current_image and current_image['intersection_points']:
            x_coords, y_coords = zip(*current_image['intersection_points'])
            self.ax.scatter(x_coords, y_coords, c='blue', s=10, label='Intersections')
        
        if 'vanishing_points' in current_image and current_image['vanishing_points']:
            x_coords, y_coords = zip(*current_image['vanishing_points'])
            self.ax.scatter(x_coords, y_coords, c='yellow', s=20, label='Vanishing Points')
        
        # Redraw manual lines if they exist
        self.line_artists = []  # Clear the list since ax.clear() removed the artists
        for line_points in self.manual_lines:
            start, end = line_points
            line_artist, = self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                                      'r-', linewidth=2, alpha=0.8)
            self.line_artists.append(line_artist)
        
        # Redraw manual vanishing points if they exist
        self.vanishing_point_artists = []  # Clear the list since ax.clear() removed the artists
        if hasattr(self, '_manual_vp_coords') and self._manual_vp_coords is not None:
            vp_artist = self.ax.scatter(self._manual_vp_coords[0], self._manual_vp_coords[1], 
                                      c='red', s=50, marker='x', linewidth=3, label='Manual VP')
            self.vanishing_point_artists.append(vp_artist)
        
        # Add legend if there are overlays or manual vanishing points
        has_overlays = ('intersection_points' in current_image and current_image['intersection_points']) or \
                      ('vanishing_points' in current_image and current_image['vanishing_points'])
        has_manual_vp = len(self.vanishing_point_artists) > 0
        
        if has_overlays or has_manual_vp:
            self.ax.legend()
        
        # Build title with drawing mode indicator
        title_parts = [f"{title} ({self.current_index + 1}/{len(self.images)})"]
        if self.drawing_mode:
            title_parts.append("DRAWING MODE")
        if self.manual_lines:
            title_parts.append(f"({len(self.manual_lines)} manual lines)")
        
        self.ax.set_title(" - ".join(title_parts))
        
        # Add grid if specified (default to True for processed images)
        show_grid = current_image.get('show_grid', True)
        if show_grid:
            self.ax.grid(True, linestyle="--", alpha=0.7, color='grey')
        
        # Keep axis on for processed images to show grid, off for original
        show_axis = current_image.get('show_axis', True)
        if not show_axis:
            self.ax.axis('off')
        
        # Update the display
        self.fig.canvas.draw()
    
    def show(self):
        """Display the image viewer with navigation controls."""
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Connect keyboard event handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Initial display
        self._update_display()
        
        # Add instructions
        instruction_text = (
            "Navigation: ← → (A/D) | Drawing: M=toggle, C=clear, V=calc VP | Q/Esc=quit"
        )
        self.fig.suptitle(instruction_text, fontsize=10, y=0.02)
        
        plt.tight_layout()
        plt.show()
