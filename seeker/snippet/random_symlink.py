#date: 2021-09-01T13:10:32Z
#url: https://api.github.com/gists/46aeccc2a0ae6d502a8b1504a8a68214
#owner: https://api.github.com/users/jkawczynski

import argparse
import os
import random
import time
import typing as T


def activate_wallpaper(wallpaper_path: str, symlink_path: str):
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(wallpaper_path, symlink_path)


def get_all_wallpapers(target_directory) -> T.List[str]:
    return os.listdir(target_directory)


def get_random_wallpaper(wallpapers: T.List[str]) -> str:
    return random.choice(wallpapers)


def rotate_wallpapers(symlink_path: str, target_directory: str, interval: int):
    while True:
        wallpapers = get_all_wallpapers(target_directory)
        wallpaper_file_name = get_random_wallpaper(wallpapers)
        wallpaper_path = os.path.join(target_directory, wallpaper_file_name)
        print(f"Activating wallpaper: {wallpaper_file_name}")
        activate_wallpaper(wallpaper_path, symlink_path)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly choose a file from upload directory, and create symlink for it"
    )
    parser.add_argument(
        "--symlink-path",
        dest="symlink_path",
        type=str,
        help="Absolute path where symlink should be created",
    )
    parser.add_argument(
        "--target-directory",
        dest="target_directory",
        type=str,
        help="Directory from which files will be randomly selected to create symlink",
    )
    parser.add_argument(
        "--interval", dest="interval", type=int, default=30, help="Interval in seconds"
    )
    args = parser.parse_args()
    rotate_wallpapers(args.symlink_path, args.target_directory, args.interval)
