#date: 2026-02-03T17:31:30Z
#url: https://api.github.com/gists/6090e22f32c09404d0034805c708671e
#owner: https://api.github.com/users/bendymind

import yt_dlp

def get_vimeo_transcript(video_url):
    ydl_opts = {
        'skip_download': True,        # We don't want the video file
        'writesubtitles': True,       # Get subtitles
        'writeautomaticsub': True,    # Get auto-generated ones if manual is missing
        'subtitleslangs': ['en.*'],   # Fetch English (regex for 'en', 'en-US', etc.)
        'outtmpl': 'transcript',      # Temporary filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Example usage
# get_vimeo_transcript('https://vimeo.com/123456789')