#date: 2025-08-22T17:11:28Z
#url: https://api.github.com/gists/99bcf1492c657683d3a30bc595377397
#owner: https://api.github.com/users/cestrus-ai

import cv2
import numpy as np

# definitions
video_source = 0
video_full = 1
video_message_limit = 60
camera_width = 720
camera_height = 480
camera_fps = 1

# state
state = {
    'bee_state' : 'OFF', # OFF, READY, ...
    'rssi':0,
    'rssi_msg':'Strong signal',
    'frame': {},
    'video_msg': '[Manual control is ON]',
    'video_msg_countdown':0
}

# Autopilot's overlay
def draw_rc_auto_status(frame):
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    
    color = color_green if state['rssi_msg'] == 'Strong signal' else color_red
    cv2.circle(frame, (50, 50), 7, color, -1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "RC", (65, 55), font, 0.5, color, 2)

    if state['bee_state'] == 'OFF':
        cv2.putText(frame, 'MANUAL', (110, 55), font, 0.5, color_red, 1)
    else:
        cv2.putText(frame, 'AUTO', (110, 55), font, 0.5, color_green, 2)

def draw_dotted_line(frame, start, end, color, thickness, gap):
    x1, y1 = start
    x2, y2 = end
    length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    for i in range(0, length, gap * 2):
        start_x = int(x1 + (x2 - x1) * i / length)
        start_y = int(y1 + (y2 - y1) * i / length)
        end_x = int(x1 + (x2 - x1) * (i + gap) / length)
        end_y = int(y1 + (y2 - y1) * (i + gap) / length)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)

def draw_cross_target(frame):
    color_white = (255, 255, 255)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    draw_dotted_line(frame, (center_x - 50, center_y), 
                     (center_x + 50, center_y), color_white, 2, 5)

    draw_dotted_line(frame, (center_x, center_y - 50), 
                     (center_x, center_y + 50), color_white, 2, 5)

def draw_scaled_target(frame):
    color_white = (255, 255, 255)
    rect_size = 50
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    top_left_x = center_x - rect_size // 2
    top_left_y = center_y - rect_size // 2

    center_region = frame[top_left_y:top_left_y + rect_size, 
                          top_left_x:top_left_x + rect_size]
    scaled_region = cv2.resize(center_region, (rect_size * 2, rect_size * 2), 
                               interpolation=cv2.INTER_LINEAR)

    overlay_x_start = width - rect_size * 2 - 20
    overlay_y_start = 20
    frame[overlay_y_start:overlay_y_start + rect_size * 2, 
          overlay_x_start:overlay_x_start + rect_size * 2] = scaled_region

    cv2.rectangle(frame, (overlay_x_start, overlay_y_start),
                (overlay_x_start + rect_size * 2, overlay_y_start + rect_size * 2), color_white, 1)

def draw_video_message(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_white = (256, 256, 256)
    
    if state['video_msg'] != '':
        cv2.putText(frame, state['video_msg'], (43, 80), font, 0.5, color_white, 1)
        
        countdown = int(state['video_msg_countdown'])
        if countdown < video_message_limit:
            state['video_msg_countdown'] = countdown + 1
        else:
            state['video_msg'] = ''
            state['video_msg_countdown'] = 0

# Main function
def main():
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, camera_fps)

    if video_full:
        cv2.namedWindow("BEE", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("BEE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Save current frame to state for
        # Computer Vision tasks
        state['frame'] = frame

        draw_rc_auto_status(frame)
        draw_scaled_target(frame)
        draw_cross_target(frame)
        draw_video_message(frame)

        cv2.imshow('BEE', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()