#date: 2025-09-23T17:02:21Z
#url: https://api.github.com/gists/7c90ad9e8432e1ae363c3e7489f7dbbf
#owner: https://api.github.com/users/ricky8955555

# Copyright 2025 Ricky8955555
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Requirements: python>=3.11.0, httpx, beautifulsoup4, lxml

"""
A scanner to scan websites that have directory index listing enabled.
"""

import asyncio
import functools
import itertools
import sys
import urllib.parse
from argparse import ArgumentParser
from asyncio import PriorityQueue, Queue, QueueShutDown
from pathlib import PosixPath
from typing import IO, Any

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

PrioritizedUrl = tuple[int, str]


def ttyprint(
    *args: Any, sep: str = " ", end: str = "\n", file: IO[str] | None = None, flush: bool = False
) -> None:
    string = sep.join(map(str, args))
    print("\033[2K\r" + string, sep="", end=end, file=file, flush=flush)


eprint = functools.partial(ttyprint, file=sys.stderr)

if sys.stdout.isatty():
    oprint = ttyprint
else:
    oprint = print


async def worker(
    in_queue: PriorityQueue[PrioritizedUrl],
    out_queue: Queue[str],
    *,
    retry: int = -1,
    delay: float = 0.5,
) -> None:
    while True:
        try:
            _, url = await in_queue.get()
        except QueueShutDown:
            break

        retry_counter = range(retry) if retry > 0 else itertools.count()

        for _ in retry_counter:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url)
            except Exception as ex:
                eprint(f"\033[1;30;43m[Warning]\033[0m Retry {url} after {delay}s for {ex!r}.")
            else:
                break
            finally:
                await asyncio.sleep(delay)
        else:
            eprint(f"\033[1;37;41m[Error]\033[0m Failed to scan {url} on max retry reached.")
            in_queue.task_done()
            continue

        soup = BeautifulSoup(resp.content, features="lxml")

        for element in soup.find_all("a"):
            if not isinstance(element, Tag):
                continue

            href = element.attrs.get("href")

            if href is None:
                continue

            if isinstance(href, str):
                href = [href]

            for link in href:
                link = urllib.parse.urljoin(url, link)

                parsed = urllib.parse.urlparse(link)
                parsed_base = urllib.parse.urlparse(url)

                path = PosixPath(parsed.path)
                base_path = PosixPath(parsed_base.path)

                depth = len(path.parts)
                base_depth = len(base_path.parts)

                if depth <= base_depth:
                    continue

                if not path.suffix and parsed.path.endswith("/"):
                    await in_queue.put((-depth, link))
                else:
                    await out_queue.put(link)

        in_queue.task_done()


async def root_worker(
    url: str,
    in_queue: PriorityQueue[PrioritizedUrl],
    out_queue: Queue[str],
    nworker: int = 16,
    **kwargs: Any,
) -> None:
    workers = [asyncio.create_task(worker(in_queue, out_queue, **kwargs)) for _ in range(nworker)]

    await in_queue.put((0, url))

    await in_queue.join()
    in_queue.shutdown()

    for it in workers:
        it.cancel()

    out_queue.shutdown()


async def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--url", "-u", help="target to scan", required=True)
    parser.add_argument(
        "--nworker", "-n", help="number of workers (default: 16)", default=16, type=int
    )
    parser.add_argument(
        "--retry", "-r", help="max retries on error (default: -1)", default=-1, type=int
    )
    parser.add_argument(
        "--delay", "-d", help="delay between requests (default: 0.5)", default=0.5, type=float
    )

    args = parser.parse_args()

    in_queue = PriorityQueue[PrioritizedUrl]()
    out_queue = Queue[str]()

    asyncio.create_task(
        root_worker(args.url, in_queue, out_queue, args.nworker, retry=args.retry, delay=args.delay)
    )

    for cnt in itertools.count():
        try:
            url = await out_queue.get()
        except QueueShutDown:
            eprint("\033[1;30;44m[Info]\033[0m All workers exited.")
            break

        oprint(url)

        eprint(f"\033[1;30;42m[{cnt} scanned / {in_queue.qsize()} tasks]\033[0m {url}", end="")


asyncio.run(main())
