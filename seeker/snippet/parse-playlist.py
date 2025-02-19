#date: 2025-02-19T16:49:00Z
#url: https://api.github.com/gists/b6631d9c3f8ba3f716781df8b7548bab
#owner: https://api.github.com/users/mtmn

#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

def parse_playlist(playlist_path):
    """
    Parse M3U/M3U8 playlist file to extract artist and song information.
    Outputs in format: Artist Name - Song Name
    """
    songs = []
    
    # Enhanced patterns for music files
    patterns = [
        # Artist - Song.mp3
        r'^(?P<artist>.+?)\s*-\s*(?P<song>.+?)\.[^.]+$',
        # Song by Artist.mp3
        r'^(?P<song>.+?)\s+by\s+(?P<artist>.+?)\.[^.]+$',
        # Artist_-_Song.mp3 or Artist_-_Song.m4a etc
        r'^(?P<artist>.+?)_-_(?P<song>.+?)\.[^.]+$',
        # ArtistName--SongName.mp3
        r'^(?P<artist>.+?)--(?P<song>.+?)\.[^.]+$'
    ]
    
    try:
        with open(playlist_path, 'r', encoding='utf-8') as f:
            current_title = None
            
            for line in f:
                line = line.strip()
                
                # Handle extended M3U directives
                if line.startswith('#EXTINF:'):
                    # Try to extract title from EXTINF line
                    parts = line.split(',', 1)
                    if len(parts) > 1:
                        current_title = parts[1]
                    continue
                elif line.startswith('#'):
                    continue
                elif not line:
                    continue
                
                # Get just the filename without the path and extension
                filename = os.path.basename(line)
                name_without_ext = os.path.splitext(filename)[0]
                
                # First try to use EXTINF title if available
                if current_title:
                    if ' - ' in current_title:
                        artist, song = current_title.split(' - ', 1)
                        songs.append({
                            'artist': artist.strip(),
                            'song': song.strip()
                        })
                        current_title = None
                        continue
                
                # If no EXTINF or couldn't parse it, try filename patterns
                song_info = None
                
                # Try each pattern until we find a match
                for pattern in patterns:
                    match = re.match(pattern, filename)
                    if match:
                        song_info = {
                            'artist': match.group('artist').strip(),
                            'song': match.group('song').strip()
                        }
                        break
                
                # If no pattern matched, try to split on the first hyphen if it exists
                if not song_info and ' - ' in name_without_ext:
                    parts = name_without_ext.split(' - ', 1)
                    if len(parts) == 2:
                        song_info = {
                            'artist': parts[0].strip(),
                            'song': parts[1].strip()
                        }
                
                # If still no match, use the whole filename as the song name
                if not song_info:
                    song_info = {
                        'artist': 'Unknown',
                        'song': name_without_ext.strip()
                    }
                
                songs.append(song_info)
                
        return songs
        
    except Exception as e:
        print(f"Error reading playlist: {e}")
        return []

def main():
    # Check if playlist file was provided as argument
    if len(sys.argv) != 2:
        print("Usage: parse-playlist playlist_file.m3u8")
        return
    
    playlist_path = sys.argv[1]
    
    if not os.path.exists(playlist_path):
        print(f"Playlist file '{playlist_path}' not found.")
        return
    
    songs = parse_playlist(playlist_path)
    
    # Print in requested format
    for song in songs:
        print(f"{song['artist']} - {song['song']}")

if __name__ == "__main__":
    main()

