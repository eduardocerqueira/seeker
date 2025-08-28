#date: 2025-08-28T17:12:40Z
#url: https://api.github.com/gists/cdb838d1195fd0e616fd768b02e3e306
#owner: https://api.github.com/users/Leixinjonaschang

import cv2
import numpy as np
import argparse
from tqdm import tqdm

def extract_motion_trail_auto(input_video_path, output_image_path, start_time, end_time, num_frames, threshold=20, scale_x=1.0, scale_y=1.0):
    """
    Extract frames from a video and create a motion trail image by automatically detecting and overlaying moving objects.
    
    Args:
        input_video_path (str): Path to the input video file
        output_image_path (str): Path to save the output image
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        num_frames (int): Number of frames to extract between start and end time
        threshold (int): Threshold for motion detection (0-255)
        scale_x (float): X scale for cropping the detected object
        scale_y (float): Y scale for cropping the detected object
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Convert time to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
        raise ValueError(f"Invalid time range. Video has {total_frames} frames ({total_frames/fps:.2f} seconds)")
    
    # Calculate frame indices to extract (evenly distributed)
    frames_to_extract = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    
    # Get median frame as background (similar to qiayuan's implementation)
    print("Calculating background...")
    median_frame = get_median_frame(cap, start_frame, end_frame)
    
    # Initialize the output image
    result_image = median_frame.copy()
    
    print(f"Extracting {num_frames} frames between {start_time:.2f}s and {end_time:.2f}s")
    
    # Process frames in reverse order (earliest frames on top)
    for i, frame_idx in enumerate(tqdm(reversed(frames_to_extract), desc="Processing frames")):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # Calculate the absolute difference to detect motion
        diff = cv2.absdiff(frame, median_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary mask
        _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the main moving object)
            c = max(contours, key=cv2.contourArea)
            
            # Only process if the contour is large enough (to avoid noise)
            if cv2.contourArea(c) > 100:
                # Calculate moments to find center of the object
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate bounding box for the object
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Expand the bounding box based on scale factors
                    new_w = int(w * scale_x)
                    new_h = int(h * scale_y)
                    new_x = max(0, cx - new_w // 2)
                    new_y = max(0, cy - new_h // 2)
                    new_x2 = min(width, new_x + new_w)
                    new_y2 = min(height, new_y + new_h)
                    
                    # Create a mask for the region of interest
                    roi_mask = np.zeros_like(mask)
                    cv2.drawContours(roi_mask, [c], -1, 255, -1)
                    
                    # Calculate a color based on the frame position
                    # Earlier frames will have a different hue
                    color_factor = 0.7 + 0.3 * (i / num_frames)
                    
                    # Apply the color tint to the frame
                    tinted_frame = frame.copy()
                    tinted_frame[:,:,0] = np.clip(tinted_frame[:,:,0] * color_factor, 0, 255)       # B
                    tinted_frame[:,:,1] = np.clip(tinted_frame[:,:,1], 0, 255)                      # G
                    tinted_frame[:,:,2] = np.clip(tinted_frame[:,:,2] * (2.0 - color_factor), 0, 255)  # R
                    
                    # Blend the tinted frame with the result
                    alpha = 0.7  # Opacity for blending
                    beta = 0.3   # Background visibility
                    
                    # Only blend where the mask is non-zero
                    for c in range(3):
                        result_image[:,:,c] = np.where(
                            roi_mask > 0, 
                            alpha * tinted_frame[:,:,c] + beta * result_image[:,:,c],
                            result_image[:,:,c]
                        )
    
    cap.release()
    
    # Save the output image
    cv2.imwrite(output_image_path, result_image)
    print(f"Motion trail image saved to {output_image_path}")

def get_median_frame(cap, start_frame, end_frame):
    """
    Calculate a median frame from random samples in the video to use as background.
    Similar to qiayuan's implementation.
    """
    # Number of frames to sample (10% of the range, but at least 10 frames)
    num_samples = max(10, int((end_frame - start_frame) / 10))
    
    # Generate random frame indices
    frame_indices = np.random.randint(start_frame, end_frame, num_samples)
    
    # Collect frames
    frames = []
    for idx in tqdm(frame_indices, desc="Sampling background frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    # Calculate median
    if frames:
        median = np.median(np.array(frames), axis=0).astype(np.uint8)
        return median
    else:
        raise ValueError("Failed to sample background frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract motion trails from video automatically")
    parser.add_argument("--input_video", required=True, help="Path to input video file")
    parser.add_argument("--output_image", required=True, help="Path to output image file")
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--end", type=float, required=True, help="End time in seconds")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to extract")
    parser.add_argument("--threshold", type=int, default=20, help="Threshold for motion detection (0-255)")
    parser.add_argument("--scale-x", type=float, default=1.2, help="X scale for cropping (default: 1.2)")
    parser.add_argument("--scale-y", type=float, default=1.2, help="Y scale for cropping (default: 1.2)")
    
    args = parser.parse_args()
    
    try:
        extract_motion_trail_auto(
            args.input_video,
            args.output_image,
            args.start,
            args.end,
            args.frames,
            args.threshold,
            args.scale_x,
            args.scale_y
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")
