#date: 2023-07-26T16:42:52Z
#url: https://api.github.com/gists/6c9debdc9c629dd41924656db7181eca
#owner: https://api.github.com/users/japa017

import cv2

# Function to open multiple video capture objects
def open_video_capture(avi_files):
    video_captures = []
    for avi_file in avi_files:
        cap = cv2.VideoCapture(avi_file)
        if not cap.isOpened():
            print(f"Error: Unable to open AVI file {avi_file}")
            return None
        video_captures.append(cap)
    return video_captures

# Function to release video capture objects
def release_video_capture(video_captures):
    for cap in video_captures:
        cap.release()

# Function to navigate through frames
def navigate_frames(video_captures, time_interval):
    frame_indices = [0] * len(video_captures)
    screen_width = 1920  # Change this to your screen width resolution
    num_videos = len(video_captures)
    window_width = screen_width // num_videos
    window_height = window_width * 3 // 4  # Maintain aspect ratio 4:3
    cv2.namedWindow("AVI Files", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AVI Files", screen_width, window_height)

    while True:
        for i, cap in enumerate(video_captures):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame from AVI file {i}")
                return

            x_offset = i * window_width
            cv2.imshow("AVI Files", frame)
            cv2.moveWindow("AVI Files", x_offset, 0)

        key = cv2.waitKeyEx(0)

        if key == 27:  # Esc key
            break
        elif key == ord('q') or key == ord('Q'):
            return
        elif key == ord('a') or key == ord('A'):
            for i in range(len(video_captures)):
                frame_indices[i] = max(0, frame_indices[i] - int(time_interval * 1000))
        elif key == ord('d') or key == ord('D'):
            for i in range(len(video_captures)):
                frame_indices[i] = min(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, frame_indices[i] + int(time_interval * 1000))

if __name__ == "__main__":
    avi_files = ["video1.avi", "video2.avi"]  # List the paths of your AVI video files here

    video_captures = open_video_capture(avi_files)
    if video_captures is None:
        exit(1)

    time_interval = 1  # Time interval in seconds to increment/decrement the frame

    print("Press 'a' key to go back, 'd' key to go forward, 'q' to quit.")
    navigate_frames(video_captures, time_interval)

    release_video_capture(video_captures)
    cv2.destroyAllWindows()
