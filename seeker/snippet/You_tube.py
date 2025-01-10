#date: 2025-01-10T17:09:09Z
#url: https://api.github.com/gists/6e0653089bbf05ab59815f4562e7205e
#owner: https://api.github.com/users/memo-py

from pytube import YouTube
from pytube.exceptions import VideoUnavailable

def download_video(url, path):
    try:
        # Create a YouTube object with custom headers
        yt = YouTube(url, on_progress_callback=None, on_complete_callback=None)

        # Select a stream (progressive video + audio in mp4 format)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

        if video_stream:
            # Download the video to the specified path
            video_stream.download(path)
            print(f"Downloaded '{yt.title}' successfully!")
        else:
            print("No available video streams found.")

    except VideoUnavailable:
        print("Video unavailable or restricted.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # URL of the YouTube video
    video_url = input("Enter YouTube video URL: ")

    # Directory to save the downloaded video
    download_path = input("Enter the path to save the video (e.g., C:/Users/Downloads): ")

    download_video(video_url, download_path)
