#date: 2026-03-12T17:27:51Z
#url: https://api.github.com/gists/e0a48d6824e3475e1fc683144d2f6997
#owner: https://api.github.com/users/buzdyk

"""Fetch and organize the Evangelion demo dataset for headmaster.

Headmaster (https://github.com/riouske/headmaster) trains binary classifier
heads on top of CLIP embeddings. Each head is defined by a folder with two
class buckets — the filesystem IS the config.

This script builds a demo workspace with three stacked binary heads that
cascade from broad to specific:

  1. eva_vs_other    — Is this an Evangelion character or not?
  2. rei_vs_not      — Is this Rei Ayanami?
  3. shinji_vs_not   — Is this Shinji Ikari?

Images are sourced from Safebooru (safebooru.org), a SFW anime image board.
~300 images are downloaded (~200 training, ~100 test).

Usage:
    python fetch_eva_dataset.py                    # writes to ./workspace
    python fetch_eva_dataset.py -o path/to/ws      # custom output dir
"""

import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1"
USER_AGENT = "headmaster-demo/1.0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# (tag query, raw subdirectory, number of images to fetch)
QUERIES = [
    # --- Training: Evangelion characters ---
    ("ayanami_rei solo", "rei", 50),
    ("ikari_shinji solo", "shinji", 50),
    ("souryuu_asuka_langley solo", "eva_other", 10),
    ("nagisa_kaworu solo", "eva_other", 10),
    ("katsuragi_misato solo", "eva_other", 10),
    ("makinami_mari_illustrious solo", "eva_other", 8),
    ("ikari_gendou solo", "eva_other", 8),
    ("akagi_ritsuko solo", "eva_other", 8),
    # --- Training: Non-Eva anime characters ---
    ("hatsune_miku solo", "non_eva", 10),
    ("uzumaki_naruto solo", "non_eva", 10),
    ("sailor_moon solo", "non_eva", 10),
    ("son_goku solo", "non_eva", 10),
    ("monkey_d._luffy solo", "non_eva", 10),
    ("pikachu solo", "non_eva", 5),
    # --- Test: Eva group/multi-character scenes ---
    ("neon_genesis_evangelion multiple_girls", "test_eva_group", 10),
    ("neon_genesis_evangelion multiple_boys", "test_eva_group", 10),
    ("ayanami_rei ikari_shinji", "test_eva_group", 10),
    ("souryuu_asuka_langley ayanami_rei", "test_eva_group", 10),
    ("neon_genesis_evangelion group", "test_eva_group", 10),
    # --- Test: Rei & Shinji variants ---
    ("ayanami_rei solo plugsuit", "test_rei_plugsuit", 10),
    ("ayanami_rei solo casual", "test_rei_casual", 5),
    ("ayanami_rei solo school_uniform", "test_rei_casual", 5),
    ("ikari_shinji solo plugsuit", "test_shinji_plugsuit", 10),
    ("ikari_shinji solo casual", "test_shinji_casual", 5),
    ("ikari_shinji solo school_uniform", "test_shinji_casual", 5),
    # --- Test: Non-Eva group scenes ---
    ("multiple_girls -neon_genesis_evangelion", "test_other_group", 15),
    ("group -neon_genesis_evangelion", "test_other_group", 15),
    ("multiple_boys -neon_genesis_evangelion", "test_other_group", 10),
]


def fetch_posts(tags: str, limit: int) -> list[dict]:
    """Fetch post metadata from Safebooru API."""
    encoded = urllib.parse.quote(tags, safe="")
    url = f"{API_URL}&tags={encoded}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().strip()
    if not body:
        return []
    data = json.loads(body)
    return data if isinstance(data, list) else []


def download_image(file_url: str, dest: Path) -> bool:
    """Download a single image. Returns True if downloaded, False if skipped."""
    if dest.exists():
        return False
    req = urllib.request.Request(file_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())
    return True


def list_images(directory: Path) -> list[Path]:
    """List image files in a directory, sorted for determinism."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def copy_images(images: list[Path], dest_dir: Path):
    """Copy images into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        dst = dest_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)


def fetch_raw(raw_dir: Path):
    """Download all images into raw subdirectories."""
    for tags, subdir, limit in QUERIES:
        dest_dir = raw_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{subdir}] Fetching {limit} posts for '{tags}'...")
        posts = fetch_posts(tags, limit)

        if not posts:
            print(f"  WARNING: no results for '{tags}'")
            continue

        downloaded = 0
        for post in posts:
            file_url = post.get("file_url", "")
            post_id = post["id"]
            ext = Path(post.get("image", "unknown.jpg")).suffix
            dest_path = dest_dir / f"{post_id}{ext}"

            if download_image(file_url, dest_path):
                downloaded += 1
                time.sleep(0.3)

        existing = len(list_images(dest_dir))
        print(f"  Downloaded {downloaded} new ({existing} total in {subdir}/)")


def organize(raw_dir: Path, output_dir: Path):
    """Organize raw images into heads/ and test/ workspace structure."""
    rei = list_images(raw_dir / "rei")
    shinji = list_images(raw_dir / "shinji")
    eva_other = list_images(raw_dir / "eva_other")
    non_eva = list_images(raw_dir / "non_eva")

    heads_dir = output_dir / "heads"
    test_dir = output_dir / "test"

    # Clean previous output
    for d in (heads_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)

    # --- Training heads ---
    all_eva = rei + shinji + eva_other

    copy_images(all_eva, heads_dir / "eva_vs_other" / "eva")
    copy_images(non_eva, heads_dir / "eva_vs_other" / "other")

    copy_images(rei, heads_dir / "rei_vs_not" / "rei")
    copy_images(shinji + eva_other + non_eva, heads_dir / "rei_vs_not" / "not_rei")

    copy_images(shinji, heads_dir / "shinji_vs_not" / "shinji")
    copy_images(rei + eva_other + non_eva, heads_dir / "shinji_vs_not" / "not_shinji")

    # --- Test images ---
    test_eva_group = list_images(raw_dir / "test_eva_group")
    test_other_group = list_images(raw_dir / "test_other_group")
    rei_suit = list_images(raw_dir / "test_rei_plugsuit")
    rei_casual = list_images(raw_dir / "test_rei_casual")
    shinji_suit = list_images(raw_dir / "test_shinji_plugsuit")
    shinji_casual = list_images(raw_dir / "test_shinji_casual")

    copy_images(
        test_eva_group + rei_suit + rei_casual + shinji_suit + shinji_casual,
        test_dir / "eva_vs_other" / "eva",
    )
    copy_images(test_other_group, test_dir / "eva_vs_other" / "other")

    copy_images(rei_suit + rei_casual, test_dir / "rei_vs_not" / "rei")
    copy_images(
        shinji_suit + shinji_casual + test_other_group,
        test_dir / "rei_vs_not" / "not_rei",
    )

    copy_images(shinji_suit + shinji_casual, test_dir / "shinji_vs_not" / "shinji")
    copy_images(
        rei_suit + rei_casual + test_other_group,
        test_dir / "shinji_vs_not" / "not_shinji",
    )

    # --- Summary ---
    for label, root in [("Training heads", heads_dir), ("Test sets", test_dir)]:
        print(f"\n{label}:")
        for head_dir in sorted(root.iterdir()):
            if not head_dir.is_dir():
                continue
            print(f"  {head_dir.name}/")
            for bucket in sorted(head_dir.iterdir()):
                if not bucket.is_dir():
                    continue
                n = len(list_images(bucket))
                print(f"    {bucket.name}: {n} images")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Evangelion demo dataset for headmaster"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("workspace"),
        help="Output directory (default: ./workspace)",
    )
    args = parser.parse_args()

    raw_dir = args.output_dir / "_raw"
    print(f"Downloading images to {raw_dir}/\n")
    fetch_raw(raw_dir)

    print(f"\nOrganizing into {args.output_dir}/")
    organize(raw_dir, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
