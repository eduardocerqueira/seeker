#date: 2024-05-02T16:59:31Z
#url: https://api.github.com/gists/4be889c448a26b7c4df669af4d496496
#owner: https://api.github.com/users/tyschacht

import subprocess
import tkinter as tk
from tkinter import filedialog
from modules import editor


def open_file() -> str:
    """Opens a file selection dialog and returns the selected file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfile()
    if not file_path:
        return None
    root.destroy()
    return file_path.name


def open_editor() -> str:
    return editor.edit(contents="")


def open_file_in_editor_and_continue(file: str) -> None:
    """Opens a file in the editor using the 'code' command and allows the user to continue editing."""
    if file:
        subprocess.run(["code", file])
    else:
        print("No file provided to open.")
