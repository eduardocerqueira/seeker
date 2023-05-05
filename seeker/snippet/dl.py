#date: 2023-05-05T16:46:13Z
#url: https://api.github.com/gists/609d645af40711447c5adde634fe4d66
#owner: https://api.github.com/users/ksamuel

"""
usage: download_bookmarks.py [-h] [--concurrency [CONCURRENCY]] [--directory DIRECTORY] bookmarks

positional arguments:
  bookmarks The path to the sqlite db file containing
            the  bookmarks. It's the places.sqlite file
            in your default profile dir.

optional arguments:
  -h, --help            show this help message and exit
  --concurrency [CONCURRENCY], -c [CONCURRENCY]
                        Max number of bookmarks to process in parallel
  --directory DIRECTORY, -d DIRECTORY
                        Directory to store the downloaded files. Will be recursively created if it doesn't exist. Otherwise,
                        a temp dir will be used.
"""

import argparse
import asyncio
import sqlite3
import sys
from asyncio.exceptions import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory

if not sys.version_info >= (3, 8):
    sys.exit("This script requires Python 3.8 or higher")

UA = (
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.1.6) Gecko/20070802 SeaMonkey/1.1.4"
)


async def download(i, total, url, directory, concurrency_limit):
    async with concurrency_limit:
        print(f"Downloading: {url} - START ({i}/{total})")
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                f'wget -o /dev/null -H -U "{UA}" -p -k -P {directory} {url}',
                stderr=asyncio.subprocess.PIPE,
            )

            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            if stderr:
                print(f"Downloading: {url} - ERROR ({i}/{total})", file=sys.stderr)
                err = stderr.decode("utf8", errors="replace")
                print(f"\n[stderr]\n{err}", file=sys.stderr)

            print(f"Downloading: {url} - DONE ({i}/{total})")
        except (TimeoutError, CancelledError):
            print(f"Downloading: {url} - TIMEOUT ({i}/{total})", file=sys.stderr)
        except Exception as e:
            print(f"Downloading: {url} - ERR...: '{e}' ({i}/{total})", file=sys.stderr)

        finally:
            if proc:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bookmarks",
        help="The path to the sqlite db file containing the bookmarks. It's the places.sqlite file in your default profile dir.",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        nargs="?",
        help="Max number of bookmarks to process in parallel",
        default=40,
    )
    parser.add_argument(
        "--directory",
        "-d",
        help="Directory to store the downloaded files. Will be recursively created if it doesn't exist. Otherwise, a temp dir will be used.",
    )

    args = parser.parse_args()

    directory = args.directory or TemporaryDirectory().name
    directory = Path(directory)
    try:
        directory.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        sys.exit(f"Error while creating the output directory: {e}")

    bookmark_file = Path(args.bookmarks)

    if not bookmark_file.is_file():
        sys.exit(f'Cannot find "{bookmark_file}"')

    if not bookmark_file.name == "places.sqlite":
        sys.exit(
            f'The bookmark file should be a "place.sqlite" file, got "{bookmark_file}"'
        )

    with sqlite3.connect(bookmark_file) as con:
        query = """
            SELECT url from moz_places, moz_bookmarks
            WHERE moz_places.id = moz_bookmarks.fk;
        """
        try:
            urls = {url for [url] in con.execute(query)}
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                sys.exit("Close Firefox before running this script")
            raise

    total = len(urls)
    print(f"Ready to process {total} bookmarks")
    print(f"Saving results in: {directory}")

    running_tasks = set()
    concurrency_limit = asyncio.Semaphore(args.concurrency)

    for i, url in enumerate(urls, 1):
        running_tasks.add(download(i, total, url, directory, concurrency_limit))

    await asyncio.wait(running_tasks)

    print(f"Results saved in: {directory}")


asyncio.run(main())
