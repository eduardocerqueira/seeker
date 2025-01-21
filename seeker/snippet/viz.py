#date: 2025-01-21T17:06:34Z
#url: https://api.github.com/gists/817f8e821684fa3e04d36168d17f2b7f
#owner: https://api.github.com/users/pzqian

import json
import cv2
import time
import os
import glob


def load_events(directory):
    # Find files in directory
    # Find the timing and keystroke files
    timing_files = glob.glob(os.path.join(directory, "timing_*.json"))
    keystroke_files = glob.glob(os.path.join(directory, "keystrokes_*.jsonl"))
    
    if not timing_files or not keystroke_files:
        raise FileNotFoundError("Could not find timing or keystroke files in directory")
    
    # Use the first matching files
    timing_file = timing_files[0]
    jsonl_file = keystroke_files[0]
    
    print(f"Loading timing data from: {timing_file}")
    try:
        with open(timing_file, 'r') as file:
            timing_data = json.load(file)
            start_time = timing_data['pre_start_time']
            print(f"Loaded start time: {start_time}")
    except FileNotFoundError:
        print(f"Timing file not found: {timing_file}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing timing JSON: {e}")
        raise

    print(f"Loading events from: {jsonl_file}")
    try:
        with open(jsonl_file, 'r') as file:
            events = []
            for line in file:
                if line.strip():  # Skip empty lines
                    event = json.loads(line)
                    events.append(event)
            
            print(f"Successfully loaded {len(events)} events")
            return events, start_time
    except FileNotFoundError:
        print(f"Events file not found: {jsonl_file}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing events JSON: {e}")
        raise


def overlay_events_on_video(video_file, events, start_time):
    print(f"Starting video overlay process with {len(events)} events")
    print(f"Video start timestamp: {start_time}")

    # Adjust timestamps
    for event in events:
        event["adjusted_time"] = event["timestamp"] - start_time
    print(f"Adjusted timestamps: {events[-1]['adjusted_time']}")

    # Open the video
    print(f"Opening video file: {video_file}")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps
    print(f"Video FPS: {fps}, Frame time: {frame_time:.3f}s")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {width}x{height}")

    # Start processing
    start_script_time = time.time()
    current_event_idx = 0
    frame_count = 0
    
    # Track active text events
    text_events = []  # List of (text, position, color, start_time) tuples

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or error reading frame")
            break

        frame_count += 1
        video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds
        
        if frame_count % 30 == 0:  # Log every 30 frames
            print(f"Processing frame {frame_count}, video time: {video_time:.2f}s, current_event_idx: {current_event_idx}")

        # Draw mouse events
        while current_event_idx < len(events) and events[current_event_idx]["adjusted_time"] <= video_time:
            event = events[current_event_idx]
            if event["type"] == "mouse_moved":
                x, y = event["x"], event["y"]
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            elif event["type"] in ["key_pressed", "key_released"]:
                try:
                    key_text = event.get('key', str(event.get('scancode', 'unknown')))
                    color = (255, 0, 0) if event["type"] == "key_pressed" else (0, 255, 0)
                    text = f"{event['type']}: {key_text}"
                    # Add new text event with its start time
                    text_events.append((text, color, video_time))
                except Exception as e:
                    print(f"Error processing key event: {e}, Event data: {event}")
            current_event_idx += 1

        # Draw all active text events
        y_offset = 50
        active_events = []
        for text, color, start_time in text_events:
            if video_time - start_time <= 1.0:  # Keep text visible for 1 second
                cv2.putText(
                    frame,
                    text,
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,  # Larger font size
                    color,
                    2,    # Thicker lines
                    cv2.LINE_AA,
                )
                y_offset += 40  # More spacing between lines
                active_events.append((text, color, start_time))
        
        # Update active text events
        text_events = active_events

        # Display the frame
        cv2.imshow("Video with Events", frame)

        # Wait for the frame duration or until the user presses 'q'
        if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord("q"):
            break

    print(f"Video processing complete. Processed {frame_count} frames and {current_event_idx} events")
    cap.release()
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play video with overlayed events.")
    parser.add_argument("directory", help="Directory containing the video and event files")
    args = parser.parse_args()

    try:
        # Find video file
        video_patterns = ["*.mp4", "*.mkv", "*.mov"]
        video_files = []
        for pattern in video_patterns:
            video_files.extend(glob.glob(os.path.join(args.directory, pattern)))
        
        if not video_files:
            raise FileNotFoundError("No video file found in directory")
        
        video_file = video_files[0]  # Use the first video file found
        print(f"Using video file: {video_file}")

        events, start_time = load_events(args.directory)
        overlay_events_on_video(video_file, events, start_time)
    except Exception as e:
        print(f"Error: {e}")
