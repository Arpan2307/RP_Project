import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Import Depth Anything specific modules
import sys
# Add the Depth-Anything repo to the Python path
sys.path.append('Depth-Anything') 
from depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage

# --- CONFIG ---
VIDEO_PATH = "test1.mp4"       
YOLO_MODEL_PATH = "yolov8l.pt" # Using the 'Large' model for accuracy
DEPTH_MODEL_NAME = "depth_anything_vitl14"

# --- FIX: Using 'MJPG' codec and '.avi' container. This is reliable. ---
# --- This file will be LARGE, but it will work. ---
# --- Use VLC Media Player on your Mac to play it. ---
OUTPUT_VIDEO_PATH = "final_output_1_share.avi"
# ---------------------------------------------------------------------

# --- Camera Calibration ---
FOCAL_LENGTH = 900
# --------------------------

# --- 3D Tracker Tuning ---
MATCHING_THRESHOLD = 10.0# Max 3D distance to match
MAX_UNSEEN_FRAMES = 30# Frames to keep a track before deleting
# -------------------------
SMOOTHING_FACTOR = 0.7
def setup_depth_model():
    """Loads the Depth Anything model and preprocessing steps."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DepthAnything.from_pretrained(f'LiheYoung/{DEPTH_MODEL_NAME}').to(device).eval()
    
    transform = Compose([
        Resize(
            width=714,
            height=714,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform, device

def get_depth_map(frame, model, transform, device):
    """Generates a depth map for a single video frame."""
    h, w = frame.shape[:2]
    frame_transformed = transform({'image': frame})['image']
    # .permute() and .float() are the fixes from before
    frame_tensor = torch.from_numpy(frame_transformed).permute(2,0,1).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        depth = model(frame_tensor)

    depth = cv2.resize(depth.squeeze().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = np.zeros(depth.shape, dtype=depth.dtype)
        
    return normalized_depth, depth # Return both normalized (0-1) and raw

def get_object_depth_and_center(raw_depth_map, box):
    """Calculates the median depth and 2D center for a given bounding box."""
    x1, y1, x2, y2 = map(int, box)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(raw_depth_map.shape[1], x2)
    y2 = min(raw_depth_map.shape[0], y2)
    
    if y1 >= y2 or x1 >= x2:
        return -1, None 

    depth_patch = raw_depth_map[y1:y2, x1:x2]
    if depth_patch.size == 0:
        return -1, None 

    #median_depth = np.median(depth_patch)
    median_depth=np.percentile(depth_patch,40)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    return median_depth, (center_x, center_y)

def unproject_to_3d(center_2d, depth, frame_shape):
    """Unprojects a 2D point to a 3D point."""
    h, w = frame_shape
    u, v = center_2d
    
    cx = w / 2
    cy = h / 2
    fx = fy = FOCAL_LENGTH 
    
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return np.array([X, -Y, Z])

def visualize_frame(frame, depth_map_normalized, tracks, model_names):
    annotated_frame = frame.copy()
    
    # --- VISUAL UPGRADE: Draw Depth Map ---
    # Invert: 1.0 (Close) -> 0.0 (Far)
    # We want Bright (1.0) to be Close.
    # MAGMA: Black=Low(Far), Bright=High(Close)
    inv_depth = 1.0 - depth_map_normalized
    depth_colormap = cv2.applyColorMap((inv_depth * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # Draw Tracks
    for track in tracks:
        if track['frames_unseen'] > 0: continue # Don't draw ghosts
        
        box = track['box']
        track_id = track['id']
        cls_id = track['cls_id']
        point_3d = track['point_3d']
        
        x1, y1, x2, y2 = map(int, box)
        cls_name = model_names[int(cls_id)]
        
        # Draw Box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Label
        label_id = f"ID: {track_id} {cls_name}"
        cv2.putText(annotated_frame, label_id, (x1, y1 - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw 3D Info
        if point_3d is not None:
            X, Y, Z = point_3d
            # Show Z (Depth) clearly
            label_dist = f"Depth: {Z:.1f}" 
            cv2.putText(annotated_frame, label_dist, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Stack frames
    h, w = annotated_frame.shape[:2]
    depth_colormap = cv2.resize(depth_colormap, (w, h))
    combined_frame = np.hstack((annotated_frame, depth_colormap))
    
    # Add Legend to Depth Map
    cv2.putText(combined_frame, "Bright = CLOSE", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_frame, "Dark = FAR", (w + 20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return combined_frame

def main():
    # --- 1. Load Models ---
    print("Loading YOLOv8 model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("Loading Depth Anything model...")
    depth_model, depth_transform, device = setup_depth_model()
    
    # --- 2. Setup Video ---
    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return
            
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- RELIABLE CODEC FIX: Use 'MJPG' and '.avi' ---
    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 
                        fps, (frame_width * 2, frame_height))
    
    if not out.isOpened():
        print(f"Error: Could not create video writer with 'MJPG' codec.")
        cap.release()
        return
    # --------------------------------------------------

    active_tracks=[]
    next_track_id=0

    print("Processing video... Press 'q' to quit.")
    
    # --- 3. Process Video Frame by Frame ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
                
        # --- Step A: Get Depth Map ---
        norm_depth, raw_depth = get_depth_map(frame, depth_model, depth_transform, device)
        
        # --- Step B: Get 2D Detections (using .predict, not .track) ---
        yolo_results = yolo_model.predict(frame,verbose=False,conf=0.30,iou=0.5)
        
        # --- Step C: 3D Tracking Logic ---
        current_detections=[]
        
        # Use .boxes (not .boxes.id) and check length
        if len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            class_ids = yolo_results[0].boxes.cls.cpu().numpy()
            
            for box,cls_id in zip(boxes, class_ids):
                # --- CRITICAL BUG FIX 1: Initialize point_3d to None ---
                point_3d = None
                
                median_depth, center_2d = get_object_depth_and_center(raw_depth, box)
                
                if center_2d is not None and median_depth != -1:
                    point_3d = unproject_to_3d(center_2d, median_depth, frame.shape[:2])
                
                # --- CRITICAL BUG FIX 2: Only add valid 3D points ---
                if point_3d is not None:
                    current_detections.append({
                        'box': box,
                        'cls_id': cls_id,
                        'point_3d': point_3d
                    })
        
        # --- This matching logic is now SAFE because current_detections only has valid 3D points ---
        # D. 3D Matching & Smoothing
        matched_track_indices = set()
        newly_added_detections_indices = set()
        
        if active_tracks and current_detections:
            for det_idx, det in enumerate(current_detections):
                best_match_track_idx = -1
                min_dist = MATCHING_THRESHOLD 

                for track_idx, track in enumerate(active_tracks):
                    if track_idx in matched_track_indices: continue
                    
                    # --- FIX 1: Weighted Distance ---
                    # Trust X/Y (pixels) more than Z (depth). 
                    # We multiply Z diff by 0.5 to be more lenient on depth jumps.
                    diff = det['point_3d'] - track['point_3d']
                    weighted_dist = np.sqrt(diff[0]**2 + diff[1]**2 + (0.3 * diff[2])**2)
                    
                    if weighted_dist < min_dist:
                        min_dist = weighted_dist
                        best_match_track_idx = track_idx
                        
                if best_match_track_idx != -1:
                    # Match Found!
                    track = active_tracks[best_match_track_idx]
                    
                    # --- FIX 2: Position Smoothing (EMA) ---
                    # New pos = (Smoothing * Old) + ((1-Smoothing) * New)
                    # This stops small objects from jumping around.
                    alpha = SMOOTHING_FACTOR
                    smoothed_point = (alpha * track['point_3d']) + ((1 - alpha) * det['point_3d'])
                    
                    track['point_3d'] = smoothed_point # Save smoothed pos
                    track['box'] = det['box']
                    track['cls_id'] = det['cls_id']
                    track['frames_unseen'] = 0
                    
                    matched_track_indices.add(best_match_track_idx)
                    newly_added_detections_indices.add(det_idx)
                
        # Add new
        for det_idx, det in enumerate(current_detections):
            if det_idx not in newly_added_detections_indices:
                active_tracks.append({
                    'id': next_track_id,
                    'box': det['box'],
                    'cls_id': det['cls_id'],
                    'point_3d': det['point_3d'],
                    'frames_unseen': 0
                })
                next_track_id += 1
                
        # Update unseen
        for track_idx, track in enumerate(active_tracks):
            if track_idx not in matched_track_indices:
                track['frames_unseen'] += 1
                
        # Remove old
        active_tracks = [t for t in active_tracks if t['frames_unseen'] <= MAX_UNSEEN_FRAMES]

        # E. Visuals
        combined_frame = visualize_frame(frame, norm_depth, active_tracks, yolo_model.names)
        out.write(combined_frame)
        cv2.imshow("Final Stable 3D Tracker", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()