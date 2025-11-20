import cv2
import torch
import numpy as np
import argparse
import sys
import os
import pyrealsense2 as rs  # Import Intel RealSense SDK

sys.path.append("./")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tracker.TSDMTrack import TSDMTracker
from tools.bbox import get_axis_aligned_bbox

# ---------------------------------------------------------
# 1. Setup Arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='TSDM Live/Simulation')
# If you have a .bag file, pass it here. If None, it looks for a USB Camera.
parser.add_argument('--bag_file', default=None, type=str, help="Path to .bag file for simulation")
args = parser.parse_args()

# ---------------------------------------------------------
# 2. RealSense Setup (Handles both Real Cam and .bag Simulation)
# ---------------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()

if args.bag_file and os.path.exists(args.bag_file):
    print(f"[INFO] loading simulation from: {args.bag_file}")
    
    # Tell config to use the file
    config.enable_device_from_file(args.bag_file, repeat_playback=True)
    
    # CRITICAL FIX: When reading from a file, DO NOT force specific
    # resolutions (like 640x480) or framerates. 
    # The pipeline will automatically use whatever format is inside the file.
    
else:
    print("[INFO] Attempting to connect to RealSense USB Camera...")
    # Only force these settings if we are using a real live camera
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
try:
    # Start the pipeline with the configuration
    pipeline_profile = pipeline.start(config)
except RuntimeError as e:
    print("[ERROR] Could not open camera or file.")
    print(f"Details: {e}")
    print("HINT: If using a .bag file, ensure it contains both Color and Depth streams.")
    sys.exit(1)

# Create an align object
# rs.align allows us to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# ---------------------------------------------------------
# 3. Initialize Tracker Variables
# ---------------------------------------------------------
SiamRes_dir = 'data_and_result/weight/modelRes.pth'
SiamMask_dir = 'data_and_result/weight/Res20.pth'
Dr_dir = 'data_and_result/weight/High-Low-two.pth.tar'

tracker = None
tracking_active = False
bbox_init = None

print("\n" + "="*50)
print(" CONTROLS:")
print("   's' : Select an object to track (Pauses stream)")
print("   'r' : Reset/Stop tracking")
print("   'q' : Quit")
print("="*50 + "\n")

try:
    while True:
        # -------------------------------------------------
        # 4. Acquire Data
        # -------------------------------------------------
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        image_rgb = np.asanyarray(color_frame.get_data())
        image_depth = np.asanyarray(aligned_depth_frame.get_data())

        # -------------------------------------------------
        # 5. Tracking Logic
        # -------------------------------------------------
        display_img = image_rgb.copy()

        if tracking_active and tracker is not None:
            # Run the TSDM tracker
            try:
                state = tracker.track(image_rgb, image_depth)
                
                # --- Extract Results ---
                region_Siam = state['region_Siam']
                region_nms = state['region_nms']
                region_Dr = state['region_Dr'] # The refined output
                
                # Helper to draw
                def draw_box(img, rect, color, label=None):
                    x, y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    if label:
                        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw boxes (Blue=Siam, Cyan=NMS, Red=Final Depth Refined)
                draw_box(display_img, region_Siam, (255, 0, 0))
                draw_box(display_img, region_nms, (0, 255, 255))
                draw_box(display_img, region_Dr, (0, 0, 255), "Target")
                
                score_text = f"Score: {state['score']:.3f}"
                cv2.putText(display_img, score_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"[WARNING] Tracking lost or error: {e}")
                tracking_active = False

        else:
            cv2.putText(display_img, "Press 's' to Select Object", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # -------------------------------------------------
        # 6. Visualization & User Input
        # -------------------------------------------------
        cv2.imshow('TSDM Real-Time Tracker', display_img)
        key = cv2.waitKey(1) & 0xFF

        # QUIT
        if key == ord('q'):
            break
        
        # RESET
        elif key == ord('r'):
            tracking_active = False
            tracker = None
            print("[INFO] Tracking Reset.")

        # SELECT OBJECT
        elif key == ord('s'):
            print("[INFO] Selection mode requested. Stream paused.")
            
            # Use OpenCV's built-in ROI selector
            # This opens a frozen window where you draw a box
            roi = cv2.selectROI('TSDM Real-Time Tracker', display_img, fromCenter=False, showCrosshair=True)
            
            # roi is (x, y, w, h)
            # If user pressed Enter without selecting, all are 0
            if roi[2] > 0 and roi[3] > 0:
                bbox_init = list(roi) # [x, y, w, h]
                print(f"[INFO] Initializing Tracker with bbox: {bbox_init}")
                
                # Initialize the Tracker Class with the NEW initial frame and bbox
                # Note: We re-instantiate the class because TSDM likely does init logic in __init__
                tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, bbox_init)
                
                tracking_active = True
            else:
                print("[INFO] Selection cancelled.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Resources released.")