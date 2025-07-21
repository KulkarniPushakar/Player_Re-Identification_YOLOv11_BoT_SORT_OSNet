# D:\Project_demo\demo5\tracker.py

import torch
import numpy as np
from boxmot import BoTSORT
from pathlib import Path
from collections import defaultdict, deque

class Tracker:
    """
    A wrapper for the BoT-SORT tracking algorithm that also handles
    velocity and trajectory history for tracked objects.
    """
    def __init__(self):
        """Initializes the tracker and history components."""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Tracker initialized on device: {self.device}")

        # Reverted to the lightweight Re-ID model for better performance
        reid_weights = Path('osnet_x0_25_msmt17.pt')

        self.object_tracker = BoTSORT(
            model_weights=reid_weights,
            device=self.device,
            fp16=torch.cuda.is_available(),
            per_class=False
        )
        
        # Store short history for velocity calculation
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
        # Store full history for drawing trajectories
        self.trajectory_history = defaultdict(list)

    def _get_centroid(self, bbox):
        """Calculates the center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=int)

    def get_velocity(self, track_id):
        """Calculates the velocity (pixels per frame) for a given track ID."""
        history = self.velocity_history[track_id]
        if len(history) < 2:
            return 0.0
        
        prev_point = history[-2]
        curr_point = history[-1]
        velocity = np.linalg.norm(curr_point - prev_point)
        return velocity

    def track(self, detections, frame):
        """Updates the tracker and records centroid history."""
        if detections.size == 0:
            return np.empty((0, 7))
            
        tracks = self.object_tracker.update(detections, frame)

        # Update both velocity and trajectory histories
        for track in tracks:
            bbox = track[:4]
            track_id = int(track[4])
            centroid = self._get_centroid(bbox)
            self.velocity_history[track_id].append(centroid)
            self.trajectory_history[track_id].append(centroid)

        return tracks