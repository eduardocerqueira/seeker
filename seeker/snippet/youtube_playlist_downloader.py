#date: 2026-01-01T17:06:18Z
#url: https://api.github.com/gists/f24bbadde4ba9241c44954b28a6747b6
#owner: https://api.github.com/users/sh13y

"""
YouTube Playlist Downloader & Zipper

- Downloads all videos from a YouTube playlist in the best available MP4 quality
- Organizes videos into a folder named after the playlist
- Zips the downloaded folder automatically

Requirements:
- Python 3
- yt-dlp
- ffmpeg (must be installed and available in PATH)

Usage:
1. Replace YOUR_PLAYLIST_URL_HERE with a YouTube playlist URL
2. Run the script
"""

import yt_dlp
import os
import shlex

# --- Configuration ---
playlist_url = 'YOUR_PLAYLIST_URL_HERE'

# --- Step 1: Extract Playlist Information ---
print("Extracting playlist information...")

ydl_info_extractor = yt_dlp.YoutubeDL({
    'skip_download': True,
    'simulate': True,
    'quiet': True
})

info = ydl_info_extractor.extract_info(playlist_url, download=False)

playlist_title = info.get('playlist_title')
if not playlist_title:
    playlist_title = info.get('title', 'downloaded_videos_playlist')
    print(f"Warning: Could not determine playlist title, using '{playlist_title}'.")

print(f"Detected playlist name: {playlist_title}")

# --- Step 2: Download the Playlist Videos ---
print(f"\nStarting download of playlist: {playlist_title}")

ydl_opts_download = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': os.path.join(playlist_title, '%(playlist_index)s - %(title)s.%(ext)s'),
    'merge_output_format': 'mp4',
    'postprocessors': [{
        'key': 'FFmpegVideoRemuxer',
        'preferedformat': 'mp4',
    }],
    'noplaylist': False,
    'yes_playlist': True,
    'quiet': False,
}

with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
    ydl.download([playlist_url])

print(f"\nPlaylist '{playlist_title}' downloaded successfully!")

# --- Step 3: Zip the Downloaded Folder ---
print("\nZipping downloaded videos...")

downloaded_folder_path = os.path.join('/content', playlist_title)
zip_file_path = os.path.join('/content', f"{playlist_title}.zip")

quoted_zip_file_path = shlex.quote(zip_file_path)
quoted_downloaded_folder_path = shlex.quote(downloaded_folder_path)

zip_command = f"zip -r {quoted_zip_file_path} {quoted_downloaded_folder_path}"
print(f"Executing shell command: {zip_command}")

result = os.system(zip_command)

if result == 0:
    print(f"\nSuccessfully zipped into '{zip_file_path}'")
else:
    print(f"\nError zipping folder. Exit code: {result}")

print("\nTask complete.")
