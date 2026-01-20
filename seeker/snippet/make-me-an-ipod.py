#date: 2026-01-20T17:08:47Z
#url: https://api.github.com/gists/7af22582eeb8c6550d207ed01ecd18ca
#owner: https://api.github.com/users/pathcl

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yt_dlp

# obtain it from https://developer.spotify.com/dashboard

SPOTIFY_CLIENT_ID = "xxxx"
SPOTIFY_CLIENT_SECRET = "**********"

def get_spotify_playlist_tracks(playlist_url):
    """Extract track info from a Spotify playlist."""
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret= "**********"
    ))

    playlist_id = playlist_url.split("/")[-1].split("?")[0]

    tracks = []
    results = sp.playlist_tracks(playlist_id)

    while results:
        for item in results['items']:
            track = item['track']
            if track:
                artist = track['artists'][0]['name']
                name = track['name']
                tracks.append(f"{artist} - {name}")
        results = sp.next(results) if results['next'] else None

    return tracks

def download_track(search_query, output_path="downloads"):
    """Search YouTube and download using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'quiet': False,
        'default_search': 'ytsearch',  # Use YouTube search
        'noplaylist': True,            # Don't download playlists
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Prefix with ytsearch: to search YouTube
        ydl.download([f"ytsearch:{search_query}"])

def main():
    playlist_url = "https://open.spotify.com/playlist/xxxxxx"

    print("Fetching playlist tracks...")
    tracks = get_spotify_playlist_tracks(playlist_url)
    print(f"Found {len(tracks)} tracks\n")

    for i, track in enumerate(tracks, 1):
        print(f"[{i}/{len(tracks)}] Downloading: {track}")
        try:
            download_track(track)
            print(f"  ✓ Done\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

if __name__ == "__main__":
    main()