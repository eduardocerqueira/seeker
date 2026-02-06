#date: 2026-02-06T17:28:49Z
#url: https://api.github.com/gists/ae79377e4a0c7f35b4925fd1a5636a8e
#owner: https://api.github.com/users/botthew

"""stream_gridder.py — Experimental multicam gridder for YouTube/Explore streams.

Purpose
- Build a 4x4 (or smaller) grid from LIVE video streams instead of NPS snapshot JPEGs.
- This is for LOCAL/VPS testing and a "control room" view (Option 2 attribution).

Approach
- Use `streamlink` to resolve a playable stream URL (HLS) for each YouTube watch URL.
- Use `ffmpeg` to grab a single JPEG frame from the live stream.
- Stitch frames into a grid with Pillow (same output format as gridder.py):
  - data/current_grid.jpg
  - data/grid_map.json

Notes
- This DOES NOT rebroadcast Explore's stream. It's a local monitoring/control-room grid.
- Requires: streamlink + ffmpeg installed in the runtime container/host.

Usage
  python3 stream_gridder.py --catalog streams_catalog.json --once
  python3 stream_gridder.py --catalog streams_catalog.json --interval 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from PIL import Image

GRID_W, GRID_H = 1920, 1080
GRID_COLS, GRID_ROWS = 4, 4
CELL_W, CELL_H = GRID_W // GRID_COLS, GRID_H // GRID_ROWS


def ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("data/frames", exist_ok=True)


def load_catalog(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def resolve_stream_url(watch_url: str) -> Optional[str]:
    """Resolve a best-available stream URL using streamlink."""
    try:
        # streamlink prints stream URL with --stream-url
        out = subprocess.check_output(
            ["streamlink", "--stream-url", watch_url, "best"],
            stderr=subprocess.DEVNULL,
            timeout=25,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def grab_frame(stream_url: str, out_path: str) -> bool:
    """Use ffmpeg to grab one frame."""
    try:
        subprocess.check_call(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                stream_url,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                out_path,
            ],
            timeout=35,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 10_000
    except Exception:
        return False


def build_grid(items: List[Dict[str, Any]], out_jpg: str = "data/current_grid.jpg", out_map: str = "data/grid_map.json") -> Dict[str, Dict[str, str]]:
    ensure_dirs()

    grid = Image.new("RGB", (GRID_W, GRID_H), (0, 0, 0))
    grid_map: Dict[str, Dict[str, str]] = {}

    for i, it in enumerate(items[: GRID_ROWS * GRID_COLS]):
        url = it.get("url")
        if not url:
            continue

        stream = resolve_stream_url(url)
        if not stream:
            continue

        frame_path = f"data/frames/{it.get('id','cam')}.jpg"
        if not grab_frame(stream, frame_path):
            continue

        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception:
            continue

        img = img.resize((CELL_W, CELL_H))
        x = (i % GRID_COLS) * CELL_W
        y = (i // GRID_COLS) * CELL_H
        grid.paste(img, (x, y))

        grid_map[str(i)] = {
            "id": str(it.get("id", "")),
            "label": str(it.get("label", "")),
            "source": str(it.get("source", "")),
            "url": str(url),
        }

    grid.save(out_jpg, quality=85)
    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(grid_map, f, indent=2)

    return grid_map


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nature RedZone stream gridder (experimental)")
    p.add_argument("--catalog", default="streams_catalog.json")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--once", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    items = load_catalog(args.catalog)

    while True:
        if items:
            build_grid(items)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
