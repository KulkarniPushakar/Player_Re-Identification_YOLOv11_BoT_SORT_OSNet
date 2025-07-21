# Player Re-Identification with YOLOv11, BoT-SORT, and Interactive Trajectory Visualization

This project implements a real-time player tracking and re-identification system for sports analytics. It uses a YOLOv11 model for object detection and the BoT-SORT algorithm for robust tracking. Key features include bounding box smoothing to reduce jitter and an interactive interface that allows users to click on a player to visualize their trajectory on the field.

![Demo GIF or Screensho<img width="1919" height="1010" alt="Screenshot 2025-07-21 224139" src="https://github.com/user-attachments/assets/7376fd38-e93a-40c5-ace2-ce27bd4da6b7" />
t](assets/demo.gif) <!-- Replace with your actual demo image or GIF -->



<img width="1919" height="1010" alt="Screenshot 2025-07-21 224139" src="https://github.com/user-attachments/assets/1ede41f0-a371-4ed0-a02c-b12cd2cfcabf" />



## ✨ Key Features

- **Player Detection:** Utilizes a custom-trained YOLOv11 model to detect players, goalkeepers, and referees.
- **Robust Tracking:** Employs BoT-SORT, a strong tracker that combines motion and appearance information for accurate re-identification even after occlusions.
- **Bounding Box Smoothing:** Implements a Savitzky-Golay filter to smooth bounding box coordinates, resulting in more stable and visually pleasing tracking.
- **Interactive Trajectory Visualization:** Click on any detected player to instantly draw and view their movement trajectory throughout the video. Click again to deselect.
- **Performance Metrics:** Displays real-time Frames Per Second (FPS) on the output video.

---

## 🛠️ Technologies Used

- **Detection:** YOLO (Ultralytics)
- **Tracking:** BoT-SORT (BoxMOT)
- **Core Libraries:**  
  - PyTorch  
  - OpenCV  
  - NumPy  
  - SciPy  

---

## 🚀 Setup and Installation

Follow these steps to get the project up and running on your local machine.

### 1. Clone the Repository

```sh
git clone https://github.com/KulkarniPushakar/Player_Re-Identification_YOLOv11_BoT_SORT_OSNet.git
cd Player_Re-Identification_YOLOv11_BoT_SORT_OSNet
```

### 2. Create a Virtual Environment (Recommended)

```sh
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content (if not already present):

```
numpy
scipy
torch
torchvision
ultralytics
boxmot
opencv-python
```

Then, install the packages:

```sh
pip install -r requirements.txt
```

### 4. Download Models

You need two pre-trained model files:

- **YOLO Detection Model:** Place your `best.pt` file in a `models/` directory.
- **Re-ID Model for BoT-SORT:** The tracker is configured to use `osnet_x0_25_msmt17.pt`. Download it and place it in the root directory of the project. You can usually find this model within the BoxMOT repository or its documentation.

Your project structure should look like this:

```
Player_re-Identification_YOLOv11_BoT_SORT/
├── models/
│   └── best.pt
├── assets/
│   └── 15sec_input_720p.mp4
├── main.py
├── tracker.py
├── yolo_detector.py
├── bbox_smoother.py
├── osnet_x0_25_msmt17.pt
└── requirements.txt
```

---

## ▶️ How to Run

1. **Configure Paths:**  
   Open `main.py` (or `yolo_detection_tracking.py`) and ensure the `MODEL_PATH`, `VIDEO_PATH`, and `OUTPUT_DIR` variables point to the correct locations.

   ```python
   # main.py
   MODEL_PATH = "models/best.pt"
   VIDEO_PATH = "assets/15sec_input_720p.mp4"
   OUTPUT_DIR = "output"
   ```

2. **Execute the Script:**  
   Run the main script from your terminal.

   ```sh
   python main.py
   ```

3. **Interact with the Application:**

   - An OpenCV window will appear showing the video processing in real-time.
   - Click on a player's bounding box to select them and display their trajectory.
   - Click on the same player again or another player to change the selection.
   - Press `q` or `Esc` to quit the application.
   - The processed video will be saved in the specified output directory.

---

## 📂 File Structure

- `main.py` or `yolo_detection_tracking.py`: The main script that orchestrates the detection, tracking, smoothing, and visualization pipeline. Handles user interaction.
- `yolo_detector.py`: A wrapper class for the YOLO model to perform object detection.
- `tracker.py`: A wrapper class for the BoT-SORT tracker that manages object IDs and histories.
- `bbox_smoother.py`: A utility class that applies a Savitzky-Golay filter to smooth bounding box coordinates over time.
- `assets/`: Directory for input videos.
- `models/`: Directory for detection model weights.
- `output/`: Default directory where the output video is saved.

---

## 📊 Flowchart

![Flowchart](assets/flowchart.png)  
*Replace this with your actual flowchart image.*
<img width="2394" height="1620" alt="_- visual selection" src="https://github.com/user-attachments/assets/b2667476-07c6-4f7c-bb3c-c707c8c36155" />


---


**GitHub:**
