# D:\Project_demo\demo5\yolo_detector.py

import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    """
    A wrapper for the YOLO object detector.
    """
    def __init__(self, model_path, confidence=0.5):
        """
        Initializes the YoloDetector.

        Args:
            model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
            confidence (float): The detection confidence threshold.
        """
        self.model = YOLO(model_path)
        self.model.fuse()  # Fuse layers for faster inference
        self.confidence = confidence
        
        # Get all class IDs except for 'ball'
        self.class_list = self.model.names
        self.target_class_ids = [k for k, v in self.class_list.items() if v != 'ball']

    def detect(self, image):
        """
        Performs object detection on an image.

        Args:
            image (np.ndarray): The input image in BGR format.

        Returns:
            np.ndarray: An array of detections in the format [x1, y1, x2, y2, conf, cls].
                        Returns an empty array if no target objects are detected.
        """
        if not self.target_class_ids:
            return np.empty((0, 6))

        # Pass the list of all target IDs to the model
        results = self.model.predict(
            image, 
            conf=self.confidence, 
            classes=self.target_class_ids, 
            verbose=False
        )
        
        detections = []
        for box in results[0].boxes:
            # Extract bounding box, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls])

        if not detections:
            return np.empty((0, 6))
            
        return np.array(detections)