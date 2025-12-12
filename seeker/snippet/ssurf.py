#date: 2025-12-12T17:00:16Z
#url: https://api.github.com/gists/d44e0d78b5713a604caee30c623f7594
#owner: https://api.github.com/users/N3rdL0rd

# /// script
# dependencies = [
#    "g4f>=6.7.1",
#    "pynput>=1.8.1",
#    "pyqt6>=6.10.1",
# ]
# ///

"""
shouldersurf: A minimalist (read: shitty and half-complete) LLM overlay that sits in the corner of your screen and feeds you answers.
Supports Wayland (tested on GNOME) through XWayland, X, and Windows. OSX users are own their own.

Usage:
- Run `uv run ssurf.py`
- Within `STARTUP_DELAY` seconds (by default 3), switch over to wherever you want the overlay to live (if you're working with different workspaces like on GNOME)
- Look in the bottom left of the screen and you'll see the text "waiting".
- Copy any text you want an answer to.
- The LLM will think for a second, and an answer will pop up in the bottom left. It disappears automatically after `ANSWER_TIMEOUT`ms, by default 7000 (7s).
"""

import sys
import os
import time
from g4f.client import Client
from collections import deque
from typing import Deque

if os.name == "posix":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

MODEL = "o4-mini" # llm model to use, check g4f's docs for a full list
QUIT_HOTKEY = "<f7>" # found this to work pretty well with firefox
ANSWER_TIMEOUT = 7000 # how long to show the answers before hiding again, in ms
SYSTEM_PROMPT = """You are a grading assistant designed to produce the SHORTEST POSSIBLE ANSWERS.
For example, if given a multiple-choice question with many answers, you will simply respond the ONE LETTER of the correct answer - no explanation.
If prompted with a short-answer question (or a question with no answer options), then just respond with a few words.
Use a maximum of 5-10 words. Use shorthand if needed."""
HISTORY_LEN = 5 # length of history to keep, good for short answer questions that reference past parts
DARK_MODE = False # turn on if you expect a dark background where this is going to be overlaid
STARTUP_DELAY = 3 # how long to wait for you to switch over to your target window before starting the app

history: Deque[str] = deque(maxlen=5)

from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from pynput import keyboard

client = Client()

class LLMWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, text: str):
        super().__init__()
        self.input_text = text

    def run(self):
        r = "failed"
        try:
            history.append(self.input_text)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": "user", "content": item} for item in history] # type: ignore
            )
            r = response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            pass
        
        self.result_ready.emit(r)

class OverlayWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.label = QLabel("waiting", self)
        
        self.label.setStyleSheet(f"""
            QLabel {{
                color: {'rgba(255, 255, 255, 0.3)' if DARK_MODE else 'rgba(0, 0, 0, 0.3)'}; 
                font-size: 12px; 
                font-family: monospace;
                background-color: transparent; 
            }}
        """)
        
        self.reset_timer = QTimer(self)
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.reset_to_waiting)

        self.clipboard = QApplication.clipboard()
        assert self.clipboard is not None
        self.clipboard.dataChanged.connect(self.on_clipboard_change)
        
        self.last_text = self.clipboard.text()
        self.worker = None

        self.update_layout()

    def update_layout(self):
        self.label.adjustSize()
        self.resize(self.label.size())
        self.position_bottom_left()

    def position_bottom_left(self):
        screen = QApplication.primaryScreen()
        if not screen: return
        
        screen_geo = screen.geometry()
        screen_height = screen_geo.height()
        
        margin = 10
        x = margin
        y = screen_height - self.height() - margin
        
        self.move(x, y)

    def on_clipboard_change(self):
        if self.clipboard is None:
            return
        current_text = self.clipboard.text()
        if current_text and current_text != self.last_text:
            self.last_text = current_text
            self.process_new_text(current_text)

    def process_new_text(self, text: str):
        self.reset_timer.stop()

        self.label.setText("thinking...")
        self.update_layout()

        if self.worker and self.worker.isRunning():
            try: self.worker.result_ready.disconnect()
            except TypeError: pass 
        
        self.worker = LLMWorker(text)
        self.worker.result_ready.connect(self.display_result)
        self.worker.start()

    def display_result(self, result):
        self.label.setText(result)
        self.update_layout()
        
        self.reset_timer.start(ANSWER_TIMEOUT)

    def reset_to_waiting(self):
        self.label.setText("waiting")
        self.update_layout()

def quit_app():
    """Callback for global hotkey"""
    print("Quitting!")
    QApplication.quit()

if __name__ == "__main__":
    time.sleep(STARTUP_DELAY)
    
    app = QApplication(sys.argv)
    
    listener = keyboard.GlobalHotKeys({QUIT_HOTKEY: quit_app})
    listener.start()

    window = OverlayWidget()
    window.show()
    
    exit_code = app.exec()
    
    listener.stop() 
    sys.exit(exit_code)