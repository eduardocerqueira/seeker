#date: 2024-08-14T18:11:17Z
#url: https://api.github.com/gists/4f6365cbf223b208845bd236f808a8b4
#owner: https://api.github.com/users/s3rgeym

#!/usr/bin/env python
# https://git-scm.com/book/ru/v2/Git-изнутри-Протоколы-передачи-данных
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import (
    AsyncIterator,
    Sequence,
    TextIO,
)

import aiohttp
import yarl

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from git_index import GitIndex

__author__ = "Sergey M"
__version__ = "1.0.0"

IMAGE_EXTENSIONS = (
    ".bmp",
    ".gif",
    ".heic",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
)

VIDEO_EXTENSIONS = (
    ".3gp",
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
)

AUDIO_EXTENSIONS = (
    ".aac",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
    ".wma",
)

DOCUMENT_EXTENSIONS = (
    # ".doc",
    # ".docx",
    # ".md",
    ".odp",
    ".ods",
    ".odt",
    ".pdf",
    ".pot",
    ".ppt",
    ".pptx",
    ".psd",
    ".rtf",
    ".ai",
    ".sketch",
    # в них пароли хранятся
    # ".txt",
    # ".xls",
    # ".xlsx",
)

FONT_EXTENSIONS = (".ttf", ".otf", ".woff", ".woff2", ".eot")

WEB_EXTENSIONS = (
    ".htm",
    ".html",
    ".css",
    ".less",
    ".scss",
    ".sass",
    ".pug",
    # иногда серверный код находит
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".vue",
    ".map",
    ".tpl",
    ".webmanifest",
    ".swf",
)

TRANSLATION_EXTENSIONS = (".po", ".mo")

EXECUTABLE_EXTENSIONS = (
    ".exe",
    ".dll",
    ".msi",
    ".apk",
    ".bin",
    ".crx",
)

DYNAMIC_CONTENT_EXTENSIONS = (
    ".php",
    ".jsp",
    ".aspx",
)

ALL_EXTENSIONS = (
    IMAGE_EXTENSIONS
    + VIDEO_EXTENSIONS
    + AUDIO_EXTENSIONS
    + DOCUMENT_EXTENSIONS
    + FONT_EXTENSIONS
    + WEB_EXTENSIONS
    + TRANSLATION_EXTENSIONS
    + EXECUTABLE_EXTENSIONS
    + DYNAMIC_CONTENT_EXTENSIONS
)

# FORCE_DOWNLOAD = ("robots.txt",)

COMMON_FILES = [
    "COMMIT_EDITMSG",
    "description",
    "FETCH_HEAD",
    "HEAD",
    "index",
    "info/exclude",
    "info/refs",
    "logs/HEAD",
    "objects/info/packs",
    "ORIG_HEAD",
    "packed-refs",
    "refs/remotes/origin/HEAD",
    # "hooks/applypatch-msg",
    # "hooks/commit-msg",
    # "hooks/fsmonitor-watchman",
    # "hooks/post-update",
    # "hooks/pre-applypatch",
    # "hooks/pre-commit",
    # "hooks/pre-merge-commit",
    # "hooks/pre-push",
    # "hooks/pre-rebase",
    # "hooks/pre-receive",
    # "hooks/prepare-commit-msg",
    # "hooks/push-to-checkout",
    # "hooks/sendemail-validate",
    # "hooks/update",
]

HTML_RE = re.compile(rb"\s*<[!a-zA-Z]")
LINK_RE = re.compile(b'<a href="([^"]+)')

HASH_RE = re.compile(rb"(?<!pack-)[a-f\d]{40}")
PACK_RE = re.compile(rb"pack-[a-f\d]{40}")
REFS_PATH_RE = re.compile(rb"refs(?:/[\w.-]+)+", re.ASCII)

OBJECTS_PATH_RE = re.compile(r"/objects/[a-f\d]{2}/[a-f\d]{38}$")

GIT_FOLDER = ".git"

# 4096kb
PROBE_SIZE = 1 << 22

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


class ANSI:
    CSI = "\x1b["
    RESET = f"{CSI}m"
    BLACK = f"{CSI}30m"
    RED = f"{CSI}31m"
    GREEN = f"{CSI}32m"
    YELLOW = f"{CSI}33m"
    BLUE = f"{CSI}34m"
    PURPLE = f"{CSI}35m"
    CYAN = f"{CSI}36m"
    WHITE = f"{CSI}37m"


class ColorHandler(logging.StreamHandler):
    _level_colors = {
        "DEBUG": ANSI.CYAN,
        "INFO": ANSI.GREEN,
        "WARNING": ANSI.RED,
        "ERROR": ANSI.RED,
        "CRITICAL": ANSI.RED,
    }

    _fmt = logging.Formatter("[ %(asctime)s ] %(levelname)8s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        message = self._fmt.format(record)
        return f"{self._level_colors[record.levelname]}{message}{ANSI.RESET}"


logger = logging.getLogger(__name__)


class NameSpace(argparse.Namespace):
    input: TextIO
    output: pathlib.Path
    git_folder: str
    workers: int
    user_agent: str
    timeout: float
    download_all: bool
    force_download: bool
    host_error: int
    probe_size: int
    verbosity: int


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.ArgumentParser, NameSpace]:
    parser = argparse.ArgumentParser(
        description="Dump exposed .git repositories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType(),
        default="-",
        help="File with URLs to process (default: standard input).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / "dumps",
        help="Directory to save downloaded files.",
    )
    parser.add_argument(
        "--git-folder",
        default=GIT_FOLDER,
        help="git folder",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=10,
        help="Number of asynchronous worker tasks.",
    )
    parser.add_argument(
        "-u",
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent for HTTP requests.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout.",
    )
    parser.add_argument(
        "-a",
        "--download-all",
        "--all",
        action="store_true",
        default=False,
        help="Download all files, including those usually skipped.",
    )
    parser.add_argument(
        "-f",
        "--force-download",
        action="store_true",
        default=False,
        help="Force download and overwrite existinbg files.",
    )
    parser.add_argument(
        "-e",
        "--host-error",
        type=int,
        default=-1,
        help="Number of maximum errors per host.",
    )
    parser.add_argument(
        "-s",
        "--probe-size",
        "--probe",
        type=int,
        default=PROBE_SIZE,
        help="Probe size limit.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="Be more verbosity.",
        action="count",
        default=0,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser, parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    parser, args = parse_args(argv)

    if not re.fullmatch(r"[^\\/:\*\?\|<>]+", args.git_folder):
        parser.error("invalid git folder name")

    logger.setLevel(
        max(logging.DEBUG, logging.WARNING - logging.DEBUG * args.verbosity)
    )

    logger.addHandler(ColorHandler())

    urls = set(map(normalize_url, filter(None, map(str.strip, args.input))))

    queue = asyncio.Queue()
    seen = set()
    host_error = Counter()
    executor = ProcessPoolExecutor()

    async with get_session(args) as session:
        await asyncio.gather(
            process_queue(queue, urls, args),
            *(
                worker(session, queue, seen, host_error, executor, args)
                for _ in range(args.workers)
            ),
        )

    for git_path, result in executor.map(
        restore_repo,
        (url2path(url.joinpath(args.git_folder), args.output) for url in urls),
    ):
        if result:
            logger.info(f"Git repo restored: {git_path}")
        else:
            logger.error(f"Can't restore git repo: {git_path}")

    logger.info("finished!")


def restore_repo(git_path: pathlib.Path) -> tuple[pathlib.Path, bool]:
    work_tree = git_path.parent
    temp_dir = tempfile.mkdtemp(dir=work_tree.parent)

    try:
        # скаченные файлы переместим во временный каталог
        for item in work_tree.iterdir():
            # Не трогаем .git
            if git_path.name != item.name:
                shutil.move(item, temp_dir)

        # выполним команду git checkout, которая восстановит файлы из
        # репозитория (и удалила бы все которых там нет)

        # можно ли использовать git stash для того чтобы спрятать скаченные
        # файлы, а потом восстановить?
        cmd = (
            f"git --git-dir={shlex.quote(str(git_path))}"
            f" --work-tree={shlex.quote(str(work_tree))} checkout ."
        )

        subprocess.check_call(cmd, shell=True)
        return git_path, True
    except subprocess.CalledProcessError:
        return git_path, False
    finally:
        # перемещаем скаченные файлы обратно
        sync_directories(temp_dir, work_tree)

        # удалим временный каталог
        shutil.rmtree(temp_dir)


def sync_directories(
    src_dir: str | pathlib.Path,
    dest_dir: str | pathlib.Path,
) -> None:
    src_dir = pathlib.Path(src_dir)
    dest_dir = pathlib.Path(dest_dir)

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    for src_path in src_dir.rglob("*"):
        relative_path = src_path.relative_to(src_dir)
        dest_path = dest_dir / relative_path

        if src_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
        else:
            if dest_path.exists():
                dest_path.unlink()
            shutil.copy2(src_path, dest_path)


@contextlib.asynccontextmanager
async def get_session(args: NameSpace) -> AsyncIterator[aiohttp.ClientSession]:
    tmt = aiohttp.ClientTimeout(total=args.timeout)
    con = aiohttp.TCPConnector(ssl=False, limit=None)
    async with aiohttp.ClientSession(connector=con, timeout=tmt) as session:
        session.headers.update(
            {
                "User-Agent": args.user_agent,
                "Accept": "*/*",
                "Accept-Language": "en-US,en",
            }
        )
        yield session


async def process_queue(
    queue: asyncio.Queue[QueueItem],
    urls: set[yarl.URL],
    args: NameSpace,
) -> None:
    for url in urls:
        # Проверим сначала на листинг директорий
        # Без слеша в конце перенаправит с 301 на адрес со слешем
        await queue.put((url, args.git_folder + "/"))

        await queue.put((url, ".gitignore"))

        for item in COMMON_FILES:
            await queue.put((url, args.git_folder + "/" + item))

    await queue.join()

    for _ in range(args.workers):
        queue.put_nowait((None, None))


QueueItem = tuple[yarl.URL | None, str | None, bool | None]


async def worker(
    session: aiohttp.ClientSession,
    queue: asyncio.Queue[QueueItem],
    seen: set[yarl.URL],
    host_error: Counter,
    executor: ProcessPoolExecutor,
    args: NameSpace,
) -> None:
    task_name = asyncio.current_task().get_name()
    logger.debug(f"task started: {task_name}")

    while True:
        try:
            base_url, path = await queue.get()

            if base_url is None:
                break

            if (
                args.host_error > 0
                and host_error[base_url.host] >= args.host_error
            ):
                logger.warning(
                    f"maximum host connection errors exceeded: {base_url.host}"
                )
                continue

            target_url = base_url.joinpath(path)

            if target_url in seen:
                logger.warning(f"already seen: {target_url}")
                continue

            logger.debug(f"get: {target_url}")
            response = await session.get(target_url, allow_redirects=False)

            seen.add(target_url)

            log_message = f"{response.status} - {response.url}"
            if response.status != 200:
                logger.warning(log_message)
                continue

            logger.info(log_message)

            contents = await response.content.read(args.probe_size)

            if await check_html(contents, response, base_url, path, queue):
                logger.debug(f"html in response: {response.url}")
                continue

            if path == ".gitignore":
                await handle_gitignore(contents, base_url, queue)
            elif path.startswith(args.git_folder):
                await handle_git(
                    contents, response, base_url, path, queue, executor, args
                )

            await save_file(contents, response, args)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning(f"connection error: {base_url.host}")
            host_error[base_url.host] += 1
        except Exception as ex:
            logger.error(ex)
        finally:
            queue.task_done()

    logger.debug(f"task finished: {task_name}")


async def check_html(
    contents: bytes,
    response: aiohttp.ClientResponse,
    base_url: yarl.URL,
    path: str,
    queue: asyncio.Queue[QueueItem],
) -> bool:
    # тут обрабатываются два случая:
    # 1. - проверка листинга
    # 2. - перехват 404-ой
    if not HTML_RE.match(contents):
        return False

    if b"<title>Index of /" in contents:
        logger.debug(f"directory listing detected: {response.url}")
        for link in parse_links(contents):
            # <a href="?C=N;O=D">Name</a>
            # <a href="?C=M;O=A">Last modified</a>
            # ...
            if "?" not in link:
                logger.debug(f"add link: {link}")
                await queue.put(
                    (
                        base_url,
                        path.rstrip("/") + "/" + link.lstrip("/"),
                    )
                )

    return True


async def handle_gitignore(
    contents: bytes,
    base_url: str,
    queue: asyncio.Queue,
) -> None:
    # https://git-scm.com/docs/gitignore/en
    lines = contents.decode(errors="ignore").splitlines()

    for item in lines:
        # символы, которые используются в шаблонах
        # https://www.php.net/manual/en/function.fnmatch.php
        if not re.fullmatch(r"[^?\[\]*]+", item):
            continue

        item = item.lstrip("/")

        if not item.lower().endswith(DYNAMIC_CONTENT_EXTENSIONS):
            await queue.put((base_url, item))


async def handle_git(
    contents: bytes,
    response: aiohttp.ClientResponse,
    base_url: str,
    path: str,
    queue: asyncio.Queue[QueueItem],
    executor: ProcessPoolExecutor,
    args: NameSpace,
) -> None:
    if path.endswith("/index"):
        for entry in GitIndex.parse(BytesIO(contents)):
            logger.debug(
                f"found entry in {response.url}: {entry.sha1} => {entry.filename}"
            )
            await queue.put(
                (
                    base_url,
                    args.git_folder + "/" + hash2path(entry.sha1),
                )
            )

            lower_filename = entry.filename.lower()

            if (
                args.download_all
                and lower_filename.endswith(DYNAMIC_CONTENT_EXTENSIONS)
            ) or (
                not args.download_all
                and lower_filename.endswith(ALL_EXTENSIONS)
            ):
                continue

            # пробуем скачать файл
            await queue.put((base_url, entry.filename.lstrip("/")))

    elif OBJECTS_PATH_RE.search(path):
        decompressed = await asyncio.get_event_loop().run_in_executor(
            executor,
            decompress,
            contents,
        )

        if decompressed.startswith((b"commit", b"tree")):
            for hash in parse_hashes(decompressed):
                logger.debug(f"found hash in {response.url}: {hash}")
                await queue.put(
                    (
                        base_url,
                        args.git_folder + "/" + hash2path(hash),
                    )
                )
    else:
        # Мне лень разбираться в куче форматов файлов
        for ref in parse_refs(contents):
            await queue.put((base_url, args.git_folder + "/" + ref))

        for hash in parse_hashes(contents):
            await queue.put((base_url, args.git_folder + "/" + hash2path(hash)))

        for pack in parse_packs(contents):
            for ext in ("pack", "idx"):
                await queue.put(
                    (
                        base_url,
                        f"{args.git_folder}/objects/pack/{pack}.{ext}",
                    )
                )


async def save_file(
    contents: bytes,
    response: aiohttp.ClientResponse,
    args: NameSpace,
) -> None:
    save_path = url2path(response.url, args.output)

    if save_path.exists() and not args.force_download:
        logger.warning(f"skip existing file: {save_path}")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("wb") as fp:
        fp.write(contents)
        async for chunk in response.content.iter_chunked(1 << 16):
            fp.write(chunk)

    logger.info(f"saved: {save_path}")


def decompress(data: bytes) -> bytes:
    # zlib.decompress не поддерживает частичную декомпрессию
    return zlib.decompressobj().decompress(data)


def parse_links(contents: bytes) -> list[str]:
    return list(map(bytes.decode, LINK_RE.findall(contents)))


def parse_refs(contents: bytes) -> list[str]:
    return list(map(bytes.decode, REFS_PATH_RE.findall(contents)))


def parse_hashes(contents: bytes) -> list[str]:
    return list(map(bytes.decode, HASH_RE.findall(contents)))


def parse_packs(contents: bytes) -> list[str]:
    return list(map(bytes.decode, PACK_RE.findall(contents)))


def hash2path(hash: str) -> str:
    return f"objects/{hash[:2]}/{hash[2:]}"


def url2path(url: yarl.URL, base_path: pathlib.Path) -> pathlib.Path:
    return base_path / url.host / url.path[1:]


def normalize_url(url: str) -> yarl.URL:
    return yarl.URL(("https://", "")["://" in url] + url)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
