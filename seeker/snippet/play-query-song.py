#date: 2026-01-12T17:20:45Z
#url: https://api.github.com/gists/4f9178c386c240c5cd37aa1306432591
#owner: https://api.github.com/users/rustyconover

#!/usr/bin/env python3
"""
Event handler that plays nyan-cat.mp3 while queries are running.
Starts playback on query_begin, stops on query_end.
Works on macOS, Linux, and Windows.
"""

import json
import os
import signal
import subprocess
import sys
import platform
import urllib.request
from pathlib import Path

# Path to the audio file
AUDIO_FILE = Path(__file__).parent / "nyan-cat.mp3"
AUDIO_URL = "https://dn720700.ca.archive.org/0/items/NyanCatoriginal/Nyan%20Cat%20%5Boriginal%5D.mp3"

# PID file location (keyed by query_id to handle concurrent queries)
if platform.system() == "Windows":
    PID_DIR = Path(os.environ.get("TEMP", "C:\\Temp")) / "duckdb-nyan"
else:
    PID_DIR = Path("/tmp/duckdb-nyan")


def ensure_audio_file() -> bool:
    """Download the audio file if it doesn't exist. Returns True if file is available."""
    if AUDIO_FILE.exists():
        return True
    try:
        urllib.request.urlretrieve(AUDIO_URL, AUDIO_FILE)
        return True
    except Exception:
        return False


def get_pid_file(query_id: int) -> Path:
    return PID_DIR / f"player-{query_id}.pid"


def get_player_command() -> list:
    """Return the appropriate audio player command for this platform."""
    system = platform.system()
    if system == "Darwin":
        return ["afplay", str(AUDIO_FILE)]
    elif system == "Linux":
        # Try common Linux audio players
        for player in ["mpv", "ffplay", "paplay", "aplay"]:
            try:
                subprocess.run(["which", player], capture_output=True, check=True)
                if player == "ffplay":
                    return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(AUDIO_FILE)]
                elif player == "mpv":
                    return ["mpv", "--no-video", "--really-quiet", str(AUDIO_FILE)]
                return [player, str(AUDIO_FILE)]
            except subprocess.CalledProcessError:
                continue
        return []
    elif system == "Windows":
        # Use PowerShell to play audio
        return [
            "powershell", "-c",
            f'(New-Object Media.SoundPlayer "{AUDIO_FILE}").PlaySync()'
        ]
    return []


def start_playback(query_id: int) -> None:
    """Start playing the audio file and save the player PID."""
    if not ensure_audio_file():
        return

    cmd = get_player_command()
    if not cmd:
        return

    PID_DIR.mkdir(exist_ok=True)

    # Start player in background
    if platform.system() == "Windows":
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Save PID so we can stop it later
    pid_file = get_pid_file(query_id)
    pid_file.write_text(str(proc.pid))


def stop_playback(query_id: int) -> None:
    """Stop the audio player for this query."""
    pid_file = get_pid_file(query_id)

    if not pid_file.exists():
        return

    try:
        pid = int(pid_file.read_text().strip())
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, ValueError, OSError):
        pass  # Process already gone or invalid PID
    finally:
        pid_file.unlink(missing_ok=True)


def main():
    event = json.load(sys.stdin)
    event_type = event.get("event")
    query_id = event.get("query_id", 0)

    if event_type == "query_begin":
        start_playback(query_id)
    elif event_type == "query_end":
        stop_playback(query_id)


if __name__ == "__main__":
    main()
