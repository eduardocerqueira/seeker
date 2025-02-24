#date: 2025-02-24T17:10:16Z
#url: https://api.github.com/gists/933943e300258faa4802eb7fe67b4621
#owner: https://api.github.com/users/idcesares

# video_to_gif.py
# -------------------------------------------------
# 🏃 This script converts any video file to an HD GIF.
# 📚 It uses the moviepy library for video processing.
# 🎨 Includes options for resizing, trimming, and optimizing.
# -------------------------------------------------

# Step 1: Install dependencies (run these in your terminal if needed)
# pip install moviepy

from moviepy.editor import VideoFileClip  # Handles video processing
import os

def video_to_gif(video_path, output_gif="output.gif", fps=15, start_time=None, end_time=None, height=720):
    """
    Converts a video file to a high-definition GIF.
    
    Parameters:
    - video_path (str): Path to the video file.
    - output_gif (str): Filename for the output GIF.
    - fps (int): Frames per second for the GIF (higher means smoother animation).
    - start_time (float): Start time in seconds for trimming (optional).
    - end_time (float): End time in seconds for trimming (optional).
    - height (int): Desired height of the output GIF (maintains aspect ratio).
    
    Returns:
    - None (Outputs a GIF file in the same directory)
    """
    # Step 2: Load the video file
    print(f"📂 Loading video file: {video_path}")
    clip = VideoFileClip(video_path)
    
    # Step 3: (Optional) Trim the video
    if start_time is not None and end_time is not None:
        print(f"✂️ Trimming video from {start_time}s to {end_time}s.")
        clip = clip.subclip(start_time, end_time)
    
    # Step 4: Resize for HD output (while keeping aspect ratio)
    if height is not None:
        print(f"🖼️ Resizing video to height {height}px (HD).")
        clip = clip.resize(height=height)
    
    # Step 5: Convert video to GIF
    print(f"🎬 Converting video to GIF: {output_gif}")
    clip.write_gif(output_gif, fps=fps, program='ffmpeg', opt='nq')
    print(f"✅ GIF saved as {output_gif}")

if __name__ == "__main__":
    # 🔧 CONFIGURATION: Customize these values if needed
    video_file = input("📥 Enter the path to the video file (e.g., video.mp4): ")
    output_gif_file = input("💾 Enter the output GIF filename (default: output.gif): ") or "output.gif"
    fps = int(input("🎞️ Enter desired FPS (default 15 for smoother playback): ") or 15)
    height = int(input("🖼️ Enter desired GIF height in pixels (default 720 for HD): ") or 720)
    
    # (Optional) Trim settings
    trim = input("✂️ Do you want to trim the video? (y/n): ").lower()
    start = end = None
    if trim == 'y':
        start = float(input("⏰ Enter start time in seconds: "))
        end = float(input("⏳ Enter end time in seconds: "))

    # Run the conversion
    video_to_gif(video_file, output_gif=output_gif_file, fps=fps, start_time=start, end_time=end, height=height)

    # 🎉 Final success message
    print(f"\n🎉 All done! Your GIF is saved as '{output_gif_file}' in {os.getcwd()}")