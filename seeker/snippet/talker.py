#date: 2023-12-22T16:50:34Z
#url: https://api.github.com/gists/1656871df6a4cf27a2ab07dcdb00a8a6
#owner: https://api.github.com/users/FlyingFathead

# requires `pyttsx3` -- install with:
# pip install -U pyttsx3

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import pyttsx3
import threading

class SpeechManager:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.stop_requested = False
        self.speech_thread = None

    def speak_text(self, text):
        self.stop_requested = False
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self.run_speech, args=(text,))
            self.speech_thread.start()

    def run_speech(self, text):
        sentences = text.split('.')
        for sentence in sentences:
            if self.stop_requested:
                break
            self.engine.say(sentence)
            self.engine.runAndWait()
        self.engine.stop()

    def stop_speaking(self):
        self.stop_requested = True

# Create the main window
root = tk.Tk()
root.title("Text-to-Speech")

# Set the background color to black and the text color to white
root.configure(bg='black')

# Create a scrolled text area
text_area = ScrolledText(root, wrap=tk.WORD, bg='black', fg='white')
text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Initialize Speech Manager
speech_manager = SpeechManager()

# Button commands
def start_speaking():
    speech_manager.speak_text(text_area.get("1.0", tk.END))

def stop_speaking():
    speech_manager.stop_speaking()

# Create buttons to control the speech
speak_button = tk.Button(root, text="🔊 Speak", command=start_speaking, bg='black', fg='white')
speak_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = tk.Button(root, text="Stop", command=stop_speaking, bg='black', fg='white')
stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()