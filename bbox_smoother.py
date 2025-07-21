# D:\Project_demo\demo5\bbox_smoother.py

import numpy as np
from scipy.signal import savgol_filter
from collections import defaultdict

class BBoxSmoother:
    """
    A class to smooth bounding box coordinates over a history of frames
    using a Savitzky-Golay filter.
    """
    def __init__(self, min_history=5, window_length=5, polyorder=2):
        """
        Initializes the BBoxSmoother.

        Args:
            min_history (int): The minimum number of points required to start smoothing.
            window_length (int): The window size for the Savitzky-Golay filter. Must be an odd integer.
            polyorder (int): The polynomial order for the filter. Must be less than window_length.
        """
        self.trajectories = defaultdict(list)
        self.min_history = min_history
        self.window_length = window_length
        self.polyorder = polyorder

    def smooth(self, track_id, box):
        """
        Adds a new box to the history and returns the smoothed version.

        Args:
            track_id (int): The track ID of the object.
            box (np.ndarray or list): The bounding box [x1, y1, x2, y2].

        Returns:
            list: The smoothed bounding box coordinates as integers.
        """
        self.trajectories[track_id].append(box)
        history = self.trajectories[track_id]

        # Start smoothing only after collecting enough data points
        if len(history) < self.min_history:
            return list(map(int, box))

        # Ensure the window_length is not greater than the history size
        current_window = min(self.window_length, len(history))
        # Window must be odd
        if current_window % 2 == 0:
            current_window -= 1
        # Polyorder must be less than window
        current_polyorder = min(self.polyorder, current_window - 1)
        if current_polyorder < 1:
            return list(map(int, box))
        
        smoothed_box = []
        history_np = np.array(history)

        for i in range(4): # For x1, y1, x2, y2
            # Extract the coordinate's history
            coords = history_np[:, i]
            # Apply the filter and take the last point
            smoothed_coord = savgol_filter(coords, current_window, current_polyorder)[-1]
            smoothed_box.append(int(smoothed_coord))
            
        return smoothed_box