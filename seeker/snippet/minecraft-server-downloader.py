#date: 2023-08-08T17:07:19Z
#url: https://api.github.com/gists/34923d3ef9749577061d66327d45a2f0
#owner: https://api.github.com/users/EEKIM10

import hashlib
import fnmatch
import sys
from pathlib import Path

import requests
import click
from tqdm import tqdm

_input = input


def input(__prompt=None) -> str:
    try:
        return _input(__prompt)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


def find_matching_versions(pattern: str, manifest: dict) -> list[dict]:
    return list(filter(lambda x: fnmatch.fnmatch(x["id"], pattern), manifest["versions"]))


def download(url: str, path: Path, version: str | None = None, sha1: str | None = None) -> Path | None:
    if path.is_dir():
        name = "server.jar" if not version else "server-%s.jar" % version
        path = path / name

    if path.exists():
        if not yesno("%s already exists. Overwrite? [y/N] " % path.name):
            return
        try:
            path.unlink()
        except PermissionError:
            print("You have insufficient permissions to delete %s." % path.resolve())

    try:
        path.touch(0o755)
    except PermissionError:
        print("Permission denied while creating file at %s." % path.resolve())
        return

    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.ConnectionError:
        print("Could not connect to %s." % url)
        return
    total_size = int(response.headers.get("content-length", 50000000))
    block_size = 4096
    print("Downloading %s..." % path.name)
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with path.open("wb") as file:
        try:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)
        except KeyboardInterrupt:
            print("Download cancelled.")
            path.unlink()
            return
        except ConnectionError:
            print("Connection error while downloading %s." % path.name)
            path.unlink()
            return
        except Exception as e:
            print("An error occurred while downloading %s." % path.name)
            print(e)
            path.unlink()
            return

    progress_bar.close()
    if sha1:
        print("Verifying download...")
        sha1_instance = hashlib.new("sha1")
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(block_size), b""):
                progress_bar.update(len(chunk))
                sha1_instance.update(chunk)
        progress_bar.close()
        if sha1_instance.hexdigest() != sha1:
            print("Download verification failed.")
            path.unlink()
            return
    return path


def yesno(prompt: str | None = None) -> bool:
    response = input(prompt)
    if not response.lower().startswith(("y", "n")):
        try:
            return yesno(prompt)
        except RecursionError:
            print("You're a moron.")
            return False

    return response.lower().startswith("y")


def select(options: list[str], prompt: str | None = None) -> str | None:
    options = list(map(lambda x: x.lower().strip(), options))
    response = input(prompt)
    if response.lower().strip() not in options:
        print("Invalid option. Try again.")
        try:
            return select(options, prompt)
        except RecursionError:
            print("You're a moron.")
            return
    return response.lower().strip()


def get_minecraft_manifest() -> dict:
    print("Fetching minecraft version manifest...")
    return requests.get("https://launchermeta.mojang.com/mc/game/version_manifest.json").json()


def get_version_manifest(version_id: str, manifest: dict) -> dict | None:
    print("Fetching version manifest for version %r..." % version_id)
    for version in manifest["versions"]:
        if version["id"] == version_id:
            return requests.get(version["url"]).json()
    print("Unknown version.")
    return


@click.command()
@click.option("--list-all", "--list", "-L", is_flag=True, help="List available versions.")
@click.option("--output", "-O", "-o", "output", type=click.Path(), default=Path.cwd(), help="Output path.")
@click.argument("version", required=False, nargs=-1)
def main(list_all: bool, output, version: str | None = None):
    version = " ".join(version) if version else None
    output = Path(output)
    manifest = get_minecraft_manifest()
    if list_all:
        for version in reversed(manifest["versions"]):
            print("\N{BULLET} " + version["id"])
        return

    if not version:
        for _version in reversed(manifest["versions"]):
            print("\N{BULLET} " + _version["id"])
        version_id = select(list(map(lambda x: x["id"], manifest["versions"])), "Select a version: ")
        if not version_id:
            return
    else:
        version_id = version
    
    version_ids = version_id.split(" ")
    for version_id in version_ids:
        if "*" in version_id:
            matches = find_matching_versions(version_id, manifest)
            if matches:
                version_ids.remove(version_id)
                version_ids.extend(list(map(lambda x: x["id"], matches)))
            else:
                print("No matches for '%s'. Omitting." % version_id)
                version_ids.remove(version_id)

    if len(version_ids) > 1:
        print("Downloading the following versions: %s" % ", ".join(version_ids))
    for version_id in version_ids:
        version_manifest = get_version_manifest(version_id, manifest)
        if not version_manifest:
            return

        downloads = version_manifest["downloads"]
        if "server" not in downloads:
            print("No server download available for this version.")
            return

        server_download = downloads["server"]
        server_download_url = server_download["url"]
        server_download_sha1 = server_download["sha1"]
        pth = download(server_download_url, output, version_id, server_download_sha1)
        if pth:
            print("Downloaded %s." % pth.name)


if __name__ == "__main__":
    main()
