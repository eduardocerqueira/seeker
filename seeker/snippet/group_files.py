#date: 2022-03-03T17:13:46Z
#url: https://api.github.com/gists/2cc1613a8eaa9dfd0f567e5b5eeb5db3
#owner: https://api.github.com/users/lukasgabriel

# group_files.py

from pathlib import WindowsPath
import os


THRESHOLD_MODE = True
THRESHOLD = 32
MAX_LEVELS = 4

SOURCE_PATH = WindowsPath("C:\\PATH\\TO\\FOLDER\\")


def subdivide(src, lv=0):
    if lv < MAX_LEVELS:
        for item in src.iterdir():
            if item.is_file():
                if not WindowsPath(f"{src}\\{item.name[0:lv+1]}\\").exists():
                    os.mkdir(f"{src}\\{item.name[0:lv+1]}")
                try:
                    os.replace(
                        item.resolve(), f"{src}\\{item.name[0:lv+1]}\\{item.name}"
                    )
                except Exception as e:
                    print(e)
        for folder in src.iterdir():
            if folder.is_dir():
                if (
                    sum(1 for item in folder.iterdir()) > THRESHOLD
                    or not THRESHOLD_MODE
                ):
                    subdivide(folder, lv + 1)


subdivide(SOURCE_PATH)
