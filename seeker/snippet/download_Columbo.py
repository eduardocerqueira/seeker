#date: 2025-04-29T16:43:51Z
#url: https://api.github.com/gists/2d58a15ded9b5cc81b3227a14875dfc3
#owner: https://api.github.com/users/woctezuma

from pathlib import Path

from uqload_dl import UQLoad
from uqload_dl.progress_bar import ProgressBar

FOLDER_NAME = "Downloads/Columbo"
FNAME = "columbo_links.txt"

WEBSITE_URL_PREFIX = "https://uqload"
FILE_EXT = ".mp4"


with (Path(FOLDER_NAME) / FNAME).open(encoding="utf8") as f:
    lines = f.readlines()

for i, line in enumerate(lines, start=1):
    elements = line.split(" ")
    if line.startswith(WEBSITE_URL_PREFIX):
        url = elements[0].strip()
        title = " ".join(elements[1:]).strip()
        output_file = (Path(FOLDER_NAME) / f"{title}{FILE_EXT}").resolve()
        if output_file.exists():
            print(f"[{i}] Skip {url} because {output_file} already exists.")
            continue
        else:
            print(f"[{i}] Downloading {url} to {output_file}")
            uqload = UQLoad(
                url=url,
                output_dir=FOLDER_NAME,
                output_file=title,
                on_progress_callback=lambda downloaded, total: ProgressBar(
                    total
                ).update(downloaded),
            )
            uqload.download()
