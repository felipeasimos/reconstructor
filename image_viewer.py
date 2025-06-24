"""
Interactive image viewer for displaying multiple images with keyboard navigation and manual line drawing.

This module provides an ImageViewer class that allows users to navigate through
a sequence of images using keyboard controls and draw lines manually on images.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import matplotlib.pyplot as plt
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
    
    def _update_display(self):
        """Update the displayed image."""
        self.ax.clear()
        
        current_image = self.images[self.current_index]
        image_data = current_image['data']
        title = current_image['title']
        cmap = current_image.get('cmap', None)
        
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
