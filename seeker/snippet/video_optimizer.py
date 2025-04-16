#date: 2025-04-16T16:57:11Z
#url: https://api.github.com/gists/41507f83f3b3b55783675f0ea9b9dc4b
#owner: https://api.github.com/users/mysCod3r

import os
import sys
import subprocess
import time
from datetime import datetime

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

def is_video_file(file_path):
    return os.path.splitext(file_path)[1].lower() in VIDEO_EXTENSIONS

def log_message(log_file, message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} {message}\n")


def process_video_file(file_path, log_file, count):
    print(f"\n[{count}] Processing video: {file_path}")

    filename = os.path.basename(file_path)
    filename_no_ext, _ = os.path.splitext(filename)
    parent_dir = os.path.dirname(file_path)

    optimized_root = os.path.join(parent_dir, "optimized")
    output_dir = os.path.join(optimized_root, filename_no_ext)

    os.makedirs(output_dir, exist_ok=True)

    output_video = os.path.join(output_dir, f"{filename_no_ext}.mp4")
    output_gif = os.path.join(output_dir, f"{filename_no_ext}_thumbnail.gif")
    output_frame = os.path.join(output_dir, f"{filename_no_ext}_first_frame.jpg")

    print(f"- Converting video -> {output_video}")
    try:
        subprocess.run([
            "ffmpeg", "-v", "quiet", "-stats", "-y", "-i", file_path,
            "-c:v", "libx264", "-b:v", "2676k",
            "-r", "30", "-c:a", "aac", "-strict", "experimental",
            output_video
        ], check=True)
    except subprocess.CalledProcessError:
        print("  ✗ Error: Video conversion failed")
        log_message(log_file, f"✗ Video conversion error: {file_path}")
        return False

    print("  ✓ Video successfully converted")

    print(f"- Creating GIF -> {output_gif}")
    try:
        subprocess.run([
            "ffmpeg", "-v", "quiet", "-stats", "-y", "-i", output_video,
            "-t", "2.68", "-vf", "fps=25,scale=200:333", "-c:v", "gif", "-b:v", "6113k", output_gif
        ], check=True)
        print("  ✓ GIF successfully created")
    except subprocess.CalledProcessError:
        print("  ✗ Error: GIF creation failed")
        log_message(log_file, f"✗ GIF creation error: {file_path}")
        return False

    print(f"- Extracting first frame -> {output_frame}")
    try:
        subprocess.run([
            "ffmpeg", "-v", "quiet", "-stats", "-y", "-i", file_path,
            "-vf", "select=eq(n\\,1)", "-vsync", "vfr", output_frame
        ], check=True)
        print("  ✓ First frame successfully extracted")
    except subprocess.CalledProcessError:
        print("  ✗ Error: First frame extraction failed")
        log_message(log_file, f"✗ First frame extraction error: {file_path}")
        return False

    return True


def find_all_video_files(root_dir):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            if is_video_file(full_path):
                all_files.append(full_path)
    return all_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python optimize_videos.py <main_folder_path>")
        sys.exit(1)

    root_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        sys.exit(1)

    log_file = os.path.join(os.getcwd(), f"video_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Directory to process: {root_dir}")
    print(f"Log file: {log_file}")

    video_files = find_all_video_files(root_dir)
    total_files = len(video_files)

    log_message(log_file, f"🎬 Found a total of {total_files} video files.")
    if total_files == 0:
        print("No video files found.")
        return

    success_count = 0
    fail_count = 0

    for idx, file_path in enumerate(video_files, start=1):
        success = process_video_file(file_path, log_file, idx)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print("\n=== Process completed ===")
    print(f"Total: {total_files}, Successful: {success_count}, Failed: {fail_count}")
    log_message(log_file, f"✅ Successful: {success_count}, ❌ Failed: {fail_count}")

if __name__ == "__main__":
    main()
