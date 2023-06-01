#date: 2023-06-01T16:55:57Z
#url: https://api.github.com/gists/319f16df68f99f9eb83b8db229e10694
#owner: https://api.github.com/users/emresvd

import cv2
import sys
import os
import shutil

def videoframes(video_path: str) -> int:
    frame_count = 0
    dir_path = video_path.replace(".mp4", "")

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    video = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(f"{dir_path}/{frame_count}.jpg", frame)
        frame_count += 1

    video.release()
    return frame_count

if __name__ == '__main__':
    video_path = sys.argv[1]
    frame_count = videoframes(video_path)
    print(f"extracted {frame_count} frames from {video_path}")
