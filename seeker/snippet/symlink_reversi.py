#date: 2023-02-24T17:03:27Z
#url: https://api.github.com/gists/d3598c15a208b5938a0500263edc7443
#owner: https://api.github.com/users/dmarx

from pathlib import Path
import re

images = list(Path('./frames').glob('*.png'))
n_frames = len(images)

def get_frame_index(fpath):
    fpath = str(fpath)
    match = re.findall('[0-9]+', fpath)
    if match:
        return int(match[0])

print(get_frame_index(images[-1]))

def new_frame_index(fpath, n=n_frames):
    curr_index = get_frame_index(fpath)
    max_frame = n-1
    delta = max_frame - curr_index
    return max_frame + delta

def symlink_reversi(frames):
    for frame in frames:
        new = new_frame_index(frame, len(frames))
        new_fpath = frame.parent / f"out_{new:05}.png"
        if (not new_fpath.exists()) and (new >= len(frames)):
            #frame.symlink_to(new_fpath)
            new_fpath.resolve().symlink_to(frame.resolve())

symlink_reversi(images)