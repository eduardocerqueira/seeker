#date: 2024-10-25T17:07:34Z
#url: https://api.github.com/gists/b0d4a11210a22b5346f7503e4cd7e102
#owner: https://api.github.com/users/anandsuraj

import os
import subprocess

def optimize_video(input_file, output_file):
    """
    Compress the video while maintaining good quality.

    Parameters:
    input_file (str): The path to the original video file.
    output_file (str): The path to the output compressed video file.

    Quality Parameters:
    - crf (Constant Rate Factor): 18-23 (lower is better quality, 23 is usually a good trade-off)
    - preset: 'medium' for standard speed and compression, or 'slow' for better quality
    """
    command = [
        'ffmpeg', 
        '-i', input_file,
        '-vcodec', 'libx264', 
        '-crf', '23',  # Adjust CRF value for quality (18-23)
        '-preset', 'medium',  # Adjust for speed vs. compression
        '-acodec', 'aac', 
        '-b:a', '192k',  # Audio bitrate
        output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Optimized: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error optimizing {input_file}: {e}")

def scan_and_optimize_videos(input_directory, output_directory):
    """
    Scan the directory for video files and optimize them.

    Parameters:
    input_directory (str): The root directory to scan for video files.
    output_directory (str): The directory to save optimized video files.
    """
    video_extensions = ['.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv']  # Add more formats as needed

    for root, _, files in os.walk(input_directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                input_file = os.path.join(root, file)
                # Create the output file path in the specified output directory
                output_file = os.path.join(output_directory, f"optimized_{file}")  
                optimize_video(input_file, output_file)

if __name__ == "__main__":
    # Set your input and output directories here
    input_directory_to_scan = ""  # Change this to your video directory
    output_directory = ""     # Change this to your desired output directory

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    scan_and_optimize_videos(input_directory_to_scan, output_directory)
