#date: 2025-12-08T17:04:00Z
#url: https://api.github.com/gists/c4811a090b6c4d73d7faf2c01d16fa6b
#owner: https://api.github.com/users/lucharo

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-whisper",
#     "sounddevice",
#     "numpy",
#     "scipy",
#     "pynput",
#     "loguru",
# ]
# ///
"""
Voice-to-text with push-to-talk hotkey.

Hold Right Command (⌘) to record, release to transcribe and paste.

Prerequisites:
    brew install ollama
    ollama pull qwen2.5:3b

Permissions:
    You grant permissions to the APP RUNNING THIS SCRIPT, not the script itself.
    - Running from Terminal.app → grant permissions to Terminal
    - Running from iTerm → grant permissions to iTerm
    - Running from VS Code → grant permissions to VS Code

Run:
    uv run voice_to_text.py
"""

import subprocess
import tempfile
import threading
import sys
import time
import argparse
import numpy as np
import sounddevice as sd
import mlx_whisper
from scipy.io import wavfile
from pynput import keyboard
from loguru import logger

# Config
SAMPLE_RATE = 16000
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
OLLAMA_MODEL = "qwen2.5:3b"  # No thinking mode, fast and capable for cleanup

# Change this to your preferred hotkey
# Options: keyboard.Key.cmd_r, keyboard.Key.alt_r, keyboard.Key.ctrl_r, keyboard.Key.f18, etc.
PUSH_TO_TALK_KEY = keyboard.Key.cmd_r  # Right Command key

CLEANUP_PROMPT_STRICT = """Clean up this transcription. Fix punctuation, remove filler words (um, uh, like, you know), fix obvious mishearings, keep the meaning intact. Output ONLY the cleaned text, nothing else:

{text}"""

CLEANUP_PROMPT_CASUAL = """Lightly clean up this transcription. Only fix punctuation and remove filler words (um, uh, like, you know). Do NOT restructure sentences or change word order. Keep the original phrasing. Output ONLY the cleaned text, nothing else:

{text}"""


def pause_media():
    """Pause media playback (Music, Spotify, etc.) and return what was paused."""
    result = subprocess.run([
        "osascript", "-e",
        '''
        set wasPaused to ""
        tell application "System Events"
            set musicRunning to (name of processes) contains "Music"
            set spotifyRunning to (name of processes) contains "Spotify"
        end tell

        if musicRunning then
            tell application "Music"
                if player state is playing then
                    pause
                    set wasPaused to "Music"
                end if
            end tell
        end if

        if spotifyRunning then
            tell application "Spotify"
                if player state is playing then
                    pause
                    if wasPaused is not "" then
                        set wasPaused to wasPaused & ",Spotify"
                    else
                        set wasPaused to "Spotify"
                    end if
                end if
            end tell
        end if

        return wasPaused
        '''
    ], capture_output=True, text=True)
    return result.stdout.strip()


def resume_media(paused_apps: str):
    """Resume media playback for apps that were actually paused."""
    if not paused_apps:
        return

    subprocess.run([
        "osascript", "-e",
        f'''
        set pausedApps to "{paused_apps}"

        if pausedApps contains "Music" then
            tell application "Music" to play
        end if

        if pausedApps contains "Spotify" then
            tell application "Spotify" to play
        end if
        '''
    ], capture_output=True)


def check_and_request_permissions():
    """Check for required permissions and open System Settings if needed."""
    logger.info("Checking permissions...")
    
    # Test accessibility by trying to use AppleScript keystroke
    test_result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to return "ok"'],
        capture_output=True,
        text=True
    )
    
    if "not allowed" in test_result.stderr.lower() or test_result.returncode != 0:
        logger.warning("Permissions needed!")
        logger.info("Grant permissions to your TERMINAL APP (Terminal, iTerm, Ghostty, VS Code, etc.)")
        
        # Open Accessibility settings
        logger.info("Opening Accessibility settings...")
        subprocess.run([
            "open", 
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ])
        
        input("Press Enter after granting Accessibility permission...")
        
        # Now open Input Monitoring settings
        logger.info("Opening Input Monitoring settings...")
        subprocess.run([
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
        ])
        
        input("Press Enter after granting Input Monitoring permission...")
        
        logger.success("Permissions granted. You may need to restart your terminal, then run this script again.")
        sys.exit(0)
    
    logger.success("Permissions OK")


class VoiceToText:
    def __init__(self, pause_music: bool = False, casual: bool = False):
        self.recording = False
        self.frames: list[np.ndarray] = []
        self.stream = None
        self.processing = False
        self.record_start = 0.0
        self.pause_music = pause_music
        self.casual = casual
        self.paused_apps = ""

    def audio_callback(self, indata, frame_count, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def start_recording(self):
        if self.recording or self.processing:
            return

        self.recording = True
        self.frames = []
        self.record_start = time.perf_counter()

        if self.pause_music:
            self.paused_apps = pause_media()

        logger.info("Recording...")
        
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        if not self.recording:
            return
        
        self.recording = False
        duration = time.perf_counter() - self.record_start
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        logger.info(f"Stopped ({duration:.1f}s)")
        
        if self.frames:
            # Process in background thread to not block hotkey listener
            threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        self.processing = True
        
        try:
            audio = np.concatenate(self.frames, axis=0)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wavfile.write(f.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
                temp_path = f.name

            # Transcribe
            logger.info("Transcribing...")
            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(temp_path, path_or_hf_repo=WHISPER_MODEL)
            raw_text = result["text"].strip()
            t1 = time.perf_counter()
            logger.info(f"Raw: {raw_text} ({t1-t0:.2f}s)")

            if not raw_text:
                logger.warning("No speech detected")
                if self.pause_music:
                    resume_media(self.paused_apps)
                return

            # Cleanup with LLM
            logger.info("Cleaning up...")
            t0 = time.perf_counter()
            prompt_template = CLEANUP_PROMPT_CASUAL if self.casual else CLEANUP_PROMPT_STRICT
            prompt = prompt_template.format(text=raw_text)
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL, prompt],
                capture_output=True,
                text=True,
            )
            cleaned_text = result.stdout.strip()
            t1 = time.perf_counter()
            logger.info(f"Clean: {cleaned_text} ({t1-t0:.2f}s)")

            # Paste at cursor
            self.paste_to_cursor(cleaned_text)
            logger.success("Pasted!")

            if self.pause_music:
                resume_media(self.paused_apps)

        finally:
            self.processing = False

    def paste_to_cursor(self, text: str) -> None:
        """Copy to clipboard, paste at cursor, then restore original clipboard."""
        # Save current clipboard
        original = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
        
        # Copy our text and paste
        subprocess.run(["pbcopy"], input=text, text=True)
        subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down'
        ])
        
        # Let paste complete, then restore original clipboard
        time.sleep(0.15)
        subprocess.run(["pbcopy"], input=original, text=True)

    def on_press(self, key):
        if key == PUSH_TO_TALK_KEY:
            self.start_recording()

    def on_release(self, key):
        if key == PUSH_TO_TALK_KEY:
            self.stop_recording()

    def run(self):
        mode = "casual" if self.casual else "strict"
        logger.info(f"Voice-to-Text (Push-to-Talk) [{mode} mode]")
        logger.info("Hold Right Command (⌘) to record, release to transcribe and paste. Ctrl+C to quit.")

        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice-to-text with push-to-talk")
    parser.add_argument(
        "--pause-music", 
        action="store_true",
        help="Pause Music/Spotify while recording, resume after paste"
    )
    parser.add_argument(
        "--casual",
        action="store_true",
        help="Light cleanup only (punctuation + filler words). Won't restructure sentences."
    )
    args = parser.parse_args()
    
    # Check ollama is available
    ollama_check = subprocess.run(["which", "ollama"], capture_output=True)
    if ollama_check.returncode != 0:
        logger.error("Ollama not found. Install with: brew install ollama")
        sys.exit(1)
    
    # Check model is pulled
    model_check = subprocess.run(
        ["ollama", "list"], 
        capture_output=True, 
        text=True
    )
    if OLLAMA_MODEL.split(":")[0] not in model_check.stdout:
        logger.error(f"Model not found. Pull with: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)
    
    check_and_request_permissions()
    
    # Preload Whisper model (downloads on first run, ~1.6GB)
    logger.info("Loading Whisper model (first run downloads ~1.6GB)...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write 1 second of silence
        silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
        wavfile.write(f.name, SAMPLE_RATE, silence)
        mlx_whisper.transcribe(f.name, path_or_hf_repo=WHISPER_MODEL)
    logger.success("Model loaded")
    
    app = VoiceToText(pause_music=args.pause_music, casual=args.casual)
    app.run()
