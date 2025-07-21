# D:\Project_demo\demo5\main.py

import cv2
import time
import os
import numpy as np
from yolo_detector import YoloDetector
from tracker import Tracker
from bbox_smoother import BBoxSmoother

# --- Configuration ---
MODEL_PATH = r"D:\Project_demo\demo5\models\best.pt"
VIDEO_PATH = r"D:\Project_demo\demo5\assets\15sec_input_720p.mp4"
OUTPUT_DIR = r"D:\Project_demo\demo5\output_2"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output_interactive_trajectory.mp4")
WINDOW_NAME = "Interactive Player Tracking"

# --- Color Mapping for Classes ---
COLOR_MAP = {
    'player': (0, 255, 0),      # Green
    'goalkeeper': (0, 255, 255), # Yellow
    'referee': (0, 0, 255),      # Red
}
DEFAULT_COLOR = (255, 255, 255) # White

# --- Mouse Click Handling ---
# This dictionary will store the state for mouse events
mouse_state = {
    'selected_track_id': None,
    'current_tracks': {}
}

def handle_mouse_click(event, x, y, flags, param):
    """Handles mouse clicks to select/deselect a player."""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_track_id = None
        # Check if the click is inside any bounding box
        for track_id, box_info in mouse_state['current_tracks'].items():
            x1, y1, x2, y2 = box_info['box']
            if x1 < x < x2 and y1 < y < y2:
                clicked_track_id = track_id
                break
        
        # Toggle selection
        if mouse_state['selected_track_id'] == clicked_track_id:
            mouse_state['selected_track_id'] = None # Deselect if clicking the same player
        else:
            mouse_state['selected_track_id'] = clicked_track_id

def main():
    # Initialize components
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.5)
    tracker = Tracker()
    smoother = BBoxSmoother()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    font = cv2.FONT_HERSHEY_SIMPLEX

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

    # Setup window and mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click)

    print("Processing video... Click on a player to see their trajectory.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        
        detections = detector.detect(frame)
        tracks = tracker.track(detections, frame)

        # Clear previous frame's tracks and update with current ones
        mouse_state['current_tracks'].clear()
        
        if tracks.size > 0:
            for track in tracks:
                track_id = int(track[4])
                box = track[:4]
                mouse_state['current_tracks'][track_id] = {'box': smoother.smooth(track_id, box)}

        # --- Drawing Logic ---
        selected_id = mouse_state['selected_track_id']

        # Draw trajectory for the selected player
        if selected_id is not None and selected_id in tracker.trajectory_history:
            trajectory = tracker.trajectory_history[selected_id]
            class_id = int(tracks[np.where(tracks[:, 4] == selected_id)][0, 6])
            class_name = detector.class_list.get(class_id, 'Unknown')
            color = COLOR_MAP.get(class_name, DEFAULT_COLOR)
            for i in range(1, len(trajectory)):
                cv2.line(frame, tuple(trajectory[i - 1]), tuple(trajectory[i]), color, 2)

        # Draw all bounding boxes
        for track in tracks:
            box = track[:4]
            track_id = int(track[4])
            class_id = int(track[6])
            
            class_name = detector.class_list.get(class_id, 'Unknown')
            color = COLOR_MAP.get(class_name, DEFAULT_COLOR)
            
            smoothed_box = smoother.smooth(track_id, box)
            x1, y1, x2, y2 = smoothed_box
            
            # Highlight the selected player's box
            thickness = 4 if track_id == selected_id else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Updated label without the velocity part
            label = f"{class_name.capitalize()} ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, color, 2)

        fps = 1 / (time.perf_counter() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), font, 0.7, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()