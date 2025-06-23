"""
Interactive image viewer for displaying multiple images with keyboard navigation.

This module provides an ImageViewer class that allows users to navigate through
a sequence of images using keyboard controls.
"""

from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class ImageViewer:
    """
    Interactive viewer for displaying multiple images with navigation controls.
    
    Supports keyboard navigation:
    - Left/Right arrows or A/D: Navigate between images
    - Q/Escape: Quit the viewer
    """
    
    def __init__(self, images: List[Dict[str, Any]]):
        """
        Initialize the image viewer.
        
        Args:
            images: List of dictionaries containing image data and metadata.
                   Each dict should have 'data', 'title', and optionally 'cmap' keys.
        """
        self.images = images
        self.current_index = 0
        self.fig = None
        self.ax = None
        
    def _on_key_press(self, event):
        """Handle keyboard events for navigation."""
        if event.key in ['right', 'd']:
            self.current_index = (self.current_index + 1) % len(self.images)
            self._update_display()
        elif event.key in ['left', 'a']:
            self.current_index = (self.current_index - 1) % len(self.images)
            self._update_display()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)
    
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
        
        # Add legend if there are overlays
        if ('intersection_points' in current_image and current_image['intersection_points']) or \
           ('vanishing_points' in current_image and current_image['vanishing_points']):
            self.ax.legend()
        
        self.ax.set_title(f"{title} ({self.current_index + 1}/{len(self.images)})")
        
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
        self.fig.suptitle(
            "Use ← → (or A/D) to navigate, Q/Esc to quit", 
            fontsize=10, 
            y=0.02
        )
        
        plt.tight_layout()
        plt.show()
