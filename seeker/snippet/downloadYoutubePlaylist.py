#date: 2025-03-25T16:58:35Z
#url: https://api.github.com/gists/aef3f8a5531ad8f7f38691487eca06ad
#owner: https://api.github.com/users/nodex4

import os
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB

#? use https://y2mate.nu/en-kKuO/ for the failed ones


# Initialize console for pretty output
console = Console()

# 1) Specify the YouTube Music playlist URL
playlist_url = "YOURMUSICPLAYLIST"

# 2) Prepare download folder ("songs" next to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
songs_dir = os.path.join(script_dir, "songs")
os.makedirs(songs_dir, exist_ok=True)

###############################################################################
# STEP A: Extract the playlist entries while ignoring error-causing videos
###############################################################################
extract_opts = {
    "quiet": True,
    "ignoreerrors": True,             # Don't stop if some entries are unavailable
    "ignore_no_formats_error": True,  # Ignore "no valid formats" errors
    "extract_flat": True,             # Faster: get a flat list of entries without full metadata
}

console.print("[cyan]Fetching playlist info...[/cyan]")
with yt_dlp.YoutubeDL(extract_opts) as ydl:
    try:
        info = ydl.extract_info(playlist_url, download=False, process=False)
    except yt_dlp.utils.DownloadError as e:
        console.print("[bold red]Error extracting playlist info, aborting.[/bold red]")
        console.print(str(e))
        info = {}

if not info or "entries" not in info:
    console.print("[red]No valid entries found in this playlist. Exiting.[/red]")
    raise SystemExit(1)

# Filter out any None entries (unavailable videos may return None)
entries = [e for e in info["entries"] if e is not None]
total_songs = len(entries)
console.print(f"[bold green]🎵 Total Songs Found: {total_songs}[/bold green]\n")

###############################################################################
# STEP B: Define yt-dlp options for downloading & post-processing to MP3
###############################################################################
download_opts = {
    "format": "bestaudio/best",
    "outtmpl": os.path.join(songs_dir, "%(title)s.%(ext)s"),
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "320",
        },
        {
            "key": "EmbedThumbnail"
        }
    ],
    "writethumbnail": True,
    "embedmetadata": True,
    "quiet": True,  # We'll handle our own logs
}

###############################################################################
# STEP C: Parallel download using ThreadPoolExecutor
###############################################################################
failed_downloads = []

def get_full_youtube_link(entry):
    """
    Safely reconstruct the full YouTube link from either
    entry['url'] or entry['id']. We prefer entry['url'] if available.
    """
    # Some entries from 'extract_flat' come with "url" like "https://www.youtube.com/watch?v=XXX"
    # If not present, build from ID: "https://www.youtube.com/watch?v=<id>"
    link = entry.get("url")
    if not link:
        vid_id = entry.get("id")
        if vid_id:
            link = f"https://www.youtube.com/watch?v={vid_id}"
        else:
            link = "Unknown URL"
    return link

def download_one(entry):
    """Download a single video from the entry using yt-dlp."""
    link = get_full_youtube_link(entry)
    # Each thread uses its own ydl instance
    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download([link])
    return link

console.print("[cyan]Starting parallel downloads (max 20 at once)...[/cyan]")

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}[/bold blue]"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
    TimeRemainingColumn(),
    console=console,
)

with progress:
    task = progress.add_task("[cyan]Downloading...[/cyan]", total=total_songs, filename="...")

    # Create a thread pool with up to 20 workers
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit each entry to the pool
        futures_map = {}
        for entry in entries:
            # Keep a mapping from future -> entry
            future = executor.submit(download_one, entry)
            futures_map[future] = entry

        # As each future completes, update the progress bar
        for future in as_completed(futures_map):
            entry = futures_map[future]
            title = entry.get("title", "Unknown Title")
            try:
                link = future.result()  # If there's an exception, it raises here
                # Mark the file as done
                progress.update(task, advance=1, filename=f"[green]{title} done[/green]")
            except Exception as e:
                # If something went wrong, log it
                failed_downloads.append(get_full_youtube_link(entry))
                progress.update(task, advance=1, filename=f"[red]{title} failed[/red]")

console.print("\n[bold green]Parallel downloads completed![/bold green]\n")

###############################################################################
# STEP D: Embed ID3 metadata & thumbnail in each MP3
###############################################################################
def embed_metadata(mp3_file, thumbnail_file, title, artist, album):
    audio = MP3(mp3_file, ID3=ID3)
    try:
        audio.add_tags()
    except:
        pass
    
    audio.tags.add(TIT2(encoding=3, text=title))
    audio.tags.add(TPE1(encoding=3, text=artist))
    audio.tags.add(TALB(encoding=3, text=album))

    if os.path.exists(thumbnail_file):
        with open(thumbnail_file, "rb") as img:
            audio.tags.add(APIC(
                encoding=3,
                mime="image/jpeg",
                type=3,  # Cover (front)
                desc="Cover",
                data=img.read()
            ))
    
    audio.save()

console.print("[cyan]Embedding metadata for downloaded songs...[/cyan]")
mp3_files = [f for f in os.listdir(songs_dir) if f.lower().endswith(".mp3")]

for idx, file_name in enumerate(mp3_files, start=1):
    song_title = file_name[:-4]  # Remove the ".mp3"
    thumbnail_path = os.path.join(songs_dir, song_title + ".jpg")  # yt-dlp typically saves .jpg
    mp3_path = os.path.join(songs_dir, file_name)
    
    embed_metadata(
        mp3_file=mp3_path,
        thumbnail_file=thumbnail_path,
        title=song_title,
        artist="Unknown Artist",
        album="YouTube Music"
    )
    console.print(f"[bold white]{idx}/{len(mp3_files)} - {song_title} metadata embedded[/bold white]")

###############################################################################
# STEP E: Report any failed downloads (links, not just IDs!)
###############################################################################
if failed_downloads:
    console.print("\n[bold red]❌ The following videos could NOT be downloaded:[/bold red]")
    for link in failed_downloads:
        console.print(f"  • [red]{link}[/red]")
else:
    console.print("[bold green]No failed downloads![/bold green]")

console.print("\n[bold green]🎉 All finished! Enjoy your music![/bold green]")
