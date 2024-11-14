#date: 2024-11-14T17:08:47Z
#url: https://api.github.com/gists/62b26908b8822a7596c6b663ee910d5f
#owner: https://api.github.com/users/cthoyt

from functools import lru_cache
from pathlib import Path

import bs4
import requests
import yt_dlp

# spoofing headers is needed otherwise we get a HTTP 403 Forbidden
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
}

BASE = Path("~/Dropbox/books/wind-and-truth").expanduser().resolve()
BASE.mkdir(exist_ok=True)


@lru_cache
def _get_soup(url: str) -> bs4.BeautifulSoup:
    res = requests.get(url, headers=HEADERS)
    soup = bs4.BeautifulSoup(res.text)
    return soup


def get_article_urls() -> list[str]:
    soup = _get_soup("https://reactormag.com/columns/wind-and-truth/latest")
    return sorted(
        {
            href.strip()
            for anchor in soup.find_all("a")
            if "read-wind-and-truth-by-brandon-sanderson"
            in (href := anchor.attrs["href"].removesuffix("#comments"))
        }
    )


def get_soundcloud_urls(url: str) -> dict[str, str]:
    soup = _get_soup(url)

    chapters = _get_names(soup)
    if chapters is None:
        print(f"No names found in {url}")
        return {}

    soundcloud_urls = [
        src for iframe in soup.find_all("iframe") if "soundcloud" in (src := iframe.attrs["src"])
    ]

    if len(chapters) != len(soundcloud_urls):
        print(
            f"mismatch in number of chapters and number of SoundCloud "
            f"URLs found: {chapters} and {soundcloud_urls}"
        )
        return {}

    return dict(zip(chapters, soundcloud_urls, strict=False))


def _get_names(soup):
    for header in soup.find_all("h2"):
        if "by Brandon Sanderson: Chapters" not in header.text:
            continue
        _, _, names = header.text.partition("Brandon Sanderson: Chapters ")
        names = names.replace(" and ", " ").replace(",", " ").split()
        return names


def download_video(url, output_path):
    # Define options for downloading
    ydl_opts = {
        "outtmpl": output_path.as_posix(),  # Output file name and directory
        "format": "bestvideo+bestaudio/best",  # Download the best video and audio available
        "merge_output_format": "m4b",  # Merge into an m4b file if needed
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "m4b",  # Convert to mp4 if necessary
            }
        ],
    }

    # Download the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    soundcloud_urls = {}
    for article_url in get_article_urls():
        soundcloud_urls.update(get_soundcloud_urls(article_url))

    for chapter, soundcloud_url in soundcloud_urls.items():
        stub = BASE.joinpath(chapter)
        if stub.with_suffix(".m4b").is_file():
            continue
        opus_path = stub.with_suffix(".opus")
        download_video(soundcloud_url, opus_path)


if __name__ == "__main__":
    main()
