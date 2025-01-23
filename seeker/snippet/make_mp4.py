#date: 2025-01-23T16:59:06Z
#url: https://api.github.com/gists/9469d8f859f49378dbba9e5d461b9b62
#owner: https://api.github.com/users/Cemu0

import os
import subprocess

print("Starting the process...")

# Directory containing the .png files
image_folder = 'output2'
# Output video file
video_file = 'timelapse_video2.mp4'
# Frame rate (frames per second)
fps = 23

# Remove the existing file if it exists
if os.path.exists(video_file):
    os.remove(video_file)
    print(f"Removed existing file: {video_file}")

print("Creating video with ffmpeg...")
# Use ffmpeg to create the video
subprocess.run([
    'ffmpeg',
    '-framerate', str(fps),
    '-pattern_type', 'glob',
    '-i', os.path.join(image_folder, '*.png'),
    '-c:v', 'libx264',
    '-preset', 'fast',  # Use 'slow' or 'veryslow' for better quality
    '-crf', '30',  # Lower CRF value for better quality
    '-pix_fmt', 'yuv420p',
    video_file
])


print("Video creation complete!")