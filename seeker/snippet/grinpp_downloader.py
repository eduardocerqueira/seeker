#date: 2021-12-23T16:41:04Z
#url: https://api.github.com/gists/38714c9b5267642d922a809e43ea0822
#owner: https://api.github.com/users/davidtavarez

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grin++ Downloader")
    parser.add_argument(
        "--node",
        dest="node",
        action="store_true",
        help="Download only Grin node (default: false)",
    )
    parser.set_defaults(node=False)
    required = parser.add_argument_group("required named arguments")
    required.add_argument(
        "-d", "--destination", help="Destination folder", required=True
    )

    args = parser.parse_args()

    p = Path(args.destination)

    if not p.exists() or not p.is_dir():
        print("ERROR: Invalid destination folder.")
        sys.exit(1)

    if not os.access(args.destination, os.W_OK):
        print("ERROR: Destination folder is not writable.")
        sys.exit(1)

    filename = ""
    download_url = ""
    assets = requests.get(
        "https://api.github.com/repos/GrinPlusPlus/GrinPlusPlus/releases/latest"
    ).json()["assets"]
    for asset in assets:
        if asset["name"].endswith(".AppImage"):
            filename = asset["name"]
            download_url = asset["browser_download_url"]
            break

    if not download_url:
        print("ERROR: Can't get download URL.")
        sys.exit(1)

    print("Downloading .AppImage file...")
    open(filename, "wb+").write(
        requests.get(download_url, allow_redirects=True).content
    )
    os.chmod(filename, 0o755)

    if args.node:
        print("Extracting Grin node...")

        stdout = subprocess.check_output(
            [
                os.path.join(os.path.dirname(os.path.realpath(__file__)), filename),
                "--appimage-extract",
            ]
        )

        print("Copying GrinNode binary...")
        shutil.copy2(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "squashfs-root/resources/app.asar.unpacked/bin/GrinNode",
            ),
            args.destination,
        )  # preserve file metadata.

        print("Copying tor...")
        shutil.copytree(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "squashfs-root/resources/app.asar.unpacked/bin/tor",
            ),
            f"{args.destination}/tor",
            dirs_exist_ok=True,
        )  # preserve file metadata.

        print("Removing squashfs-root folder...")
        shutil.rmtree("squashfs-root")
        os.remove(filename)

        print("Assigning execution permissions...")
        os.chmod(f"{args.destination}/GrinNode", 0o755)
        os.chmod(f"{args.destination}/tor/tor", 0o755)
    else:
        shutil.copy2(filename, args.destination)

    print("Done.")
