# Related Projects / Upstream Repositories

This repository references and is inspired by the following open-source projects:

---

### 1. Track Anything
🔗 https://github.com/gaomingqi/Track-Anything

### 2. SAM 2 (Segment Anything Model v2)
🔗 https://github.com/facebookresearch/sam2

### 3. Depth Anything V2
🔗 https://github.com/DepthAnything/Depth-Anything-V2

### 4. TSDM: Token Flow for Segment Anything Tracking
🔗 https://github.com/lql-team/TSDM

---

These repositories serve as sources of ideas, inspiration, or supporting components for this project.

## Our Addition

# Enhanced 3D Object Tracker

A computer vision pipeline that combines **YOLOv8** (for 2D object detection) and **Depth Anything** (for Monocular Depth Estimation) to track objects in 3D space $(x, y, z)$ from a single video source.

## 📝 Overview

This project takes a standard 2D video and performs the following:

1.  **Detection:** Identifies objects using YOLOv8 (Large model).
2.  **Depth Estimation:** Generates a pixel-perfect depth map using the `Depth-Anything` ViT-L model.
3.  **3D Unprojection:** Converts 2D bounding box centers + depth values into 3D coordinates using camera focal length.
4.  **Tracking:** Implements a custom greedy matching algorithm with Euclidean distance weighting and Position Smoothing (EMA) to maintain stable IDs across frames.

## 🚀 Features

  * **Hybrid AI:** Combines state-of-the-art detection and depth estimation.
  * **3D Coordinate System:** Calculates relative depth ($Z$) and lateral positions ($X, Y$).
  * **Jitter Reduction:** Uses an Exponential Moving Average (smoothing factor) to stabilize 3D coordinate readings.
  * **Dual Visualization:** Outputs a side-by-side video showing the tracked bounding boxes and the raw depth map (Magma colormap).
  * **Robust Video Saving:** Configured with `MJPG` codec in an `.avi` container to ensure compatibility across Windows and macOS.

## 🛠️ Prerequisites

  * Python 3.8+
  * CUDA-enabled GPU (Highly recommended for reasonable FPS). Running on CPU will be significantly slower.

## 📦 Installation

### 1\. Clone Project & Install Dependencies

Install the required Python libraries:

```bash
pip install opencv-python torch torchvision numpy ultralytics
```

### 2\. Install "Depth Anything"

The script relies on the `Depth-Anything` repository. You must clone it **inside** your project directory so the script can import it.

```bash
# Run this inside the folder where enhanced_3d_tracker.py is located
git clone https://github.com/LiheYoung/Depth-Anything
```

*Note: The script uses `sys.path.append('Depth-Anything')` to access the model architecture.*

## ⚙️ Configuration

Open `enhanced_3d_tracker.py` and adjust the **CONFIG** section at the top to match your setup:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `VIDEO_PATH` | Path to your input video file. | `"test1.mp4"` |
| `OUTPUT_VIDEO_PATH` | Name of the saved result file. | `"final_output_1_share.avi"` |
| `FOCAL_LENGTH` | Approximate focal length of the camera (in pixels). | `900` |
| `SMOOTHING_FACTOR` | Controls tracking stability. Higher = smoother but more lag. | `0.7` |
| `MATCHING_THRESHOLD` | Max 3D distance to consider two objects the same ID. | `10.0` |

## ▶️ Execution

1.  Place your input video (e.g., `test1.mp4`) in the project directory.
2.  Run the script:

<!-- end list -->

```bash
python enhanced_3d_tracker.py
```

**First Run Note:** The script will automatically download:

1.  The YOLOv8 weights (`yolov8l.pt`).
2.  The Depth Anything weights (via Hugging Face).
    This may take a few minutes depending on your internet connection.

## 🕹️ Controls & Output

  * **Runtime:** A window titled "Final Stable 3D Tracker" will appear showing the processing live.
  * **Quit:** Press `q` to stop processing early.
  * **Output File:** The processed video is saved as an `.avi` file (default: `final_output_1_share.avi`).

## 🧠 How it Works

1.  **Preprocessing:** The image is resized and normalized for the Depth Anything transformer.
2.  **Depth Extraction:** The model returns a raw depth map. We calculate the **median depth** (actually the 40th percentile to avoid background noise) within the bounding box of every object detected by YOLO.
3.  **3D Projection:** Using the Pinhole Camera Model:
    $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}$$
4.  **Matching Logic:**
      * The script calculates the weighted distance between existing tracks and new detections.
      * It prioritizes $X$ and $Y$ (pixel location) over $Z$ (depth) to prevent depth flickering from breaking tracks.
      * `weighted_dist = sqrt(dx² + dy² + (0.3 * dz)²)`

## ⚠️ Troubleshooting

  * **`ModuleNotFoundError: No module named 'depth_anything'`**:
      * Ensure you ran the `git clone` command in step 2 of Installation. The folder `Depth-Anything` must exist next to your `.py` file.
  * **Video Write Error**:
      * The script uses the `MJPG` codec. If the output file is empty, ensure you have write permissions in the folder.
  * **Low FPS**:
      * The "Large" (ViT-L) depth model is heavy. If you are on a CPU or weaker GPU, change `DEPTH_MODEL_NAME` in the code to `depth_anything_vits14` (Small) for better speed.


