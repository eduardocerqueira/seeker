#date: 2025-07-22T17:13:24Z
#url: https://api.github.com/gists/f4aada00178da2738dc851134e177f02
#owner: https://api.github.com/users/rubenhortas

#!/usr/bin/env python3

"""
A very lightweight Python script to download torrent files (less than X days old) from the https://showrss.info feed.
"""

import os
import re
import signal
import sys
from datetime import datetime, timedelta
from time import mktime
from types import FrameType
from typing import Optional

import feedparser

# Get your feed ulr from: https://showrss.info/feeds
# Leave the "Link type" option as "Use magnets" in feed (recommended).
USER_URL = 'https://showrss.info/user/159273.rss?magnets=true&namespaces=true&name=null&quality=hd&re=yes'

# Set the path where the torrent files wil be created.
TORRENTS_PATH = '/home/ruben/PycharmProjects/showtime/torrents'

# Number of days back from the current date to search for torrents.
THRESHOLD_DAYS = 7


def _handle_sigint(signal: int, frame: FrameType) -> None:
    exit(1)


class Cache:
    _SEPARATOR = ','
    _DATE_FORMAT = '%Y-%m-%d'

    _file_name = 'showtime.cache'
    _downloads: set[str] = set()

    def __init__(self):
        self._file = os.path.join(os.path.dirname(os.path.abspath(__file__)), self._file_name)
        self._threshold_date = (datetime.today() - timedelta(days=THRESHOLD_DAYS))
        self._get_downloads()

    def is_new(self, published_date: datetime, file_name: str) -> bool:
        entry = self._create_entry(published_date, file_name)
        return published_date >= self._threshold_date and entry not in self._downloads

    def add(self, published_date: datetime, file_name: str) -> None:
        entry = self._create_entry(published_date, file_name)
        self._downloads.add(entry)

    def write(self) -> None:
        with open(self._file, 'w') as f:
            f.write('\n'.join(sorted(list(self._downloads), reverse=True)) + '\n')

    def _create_entry(self, published_date: datetime, file_name: str) -> str:
        return f'{published_date.strftime(self._DATE_FORMAT)}{self._SEPARATOR}{file_name}'

    def _get_downloads(self) -> None:
        if os.path.isfile(self._file):

            with open(self._file) as f:
                for line in f:
                    published_date = datetime.strptime(line.split(self._SEPARATOR, 1)[0], self._DATE_FORMAT)

                    if published_date >= self._threshold_date:
                        self._downloads.add(line.strip())
                    else:
                        break


class Torrent:
    published_date: Optional[datetime]
    file_name: str = None

    _title: str = None
    _magnet_uri: str = None

    def __init__(self, title: str, date: str, magnet_uri: str):
        self._title = title
        self._magnet_uri = magnet_uri
        self.file_name = f'{title}.torrent'
        self.published_date = datetime.fromtimestamp(mktime(date))

    def __str__(self):
        return self.file_name

    def save(self) -> bool:
        # Extract the hash info (BTIH) from the magnet link
        match = re.search(r'xt=urn:btih:([^&/]+)', self._magnet_uri)

        if match:
            # Create the contents of the .torrent file in Bencode format.
            # Bencode format: d10:magnet-uri[uri_length]:[uri]e
            data = f"d10:magnet-uri{len(self._magnet_uri)}:{self._magnet_uri}e"

            try:
                file = os.path.join(TORRENTS_PATH, self.file_name)

                with open(file, 'w') as f:
                    f.write(data)

                return True
            except IOError:
                pass

        return False


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        print('Downloading torrents...')
        os.makedirs(TORRENTS_PATH, exist_ok=True)

        cache = Cache()
        feed = feedparser.parse(USER_URL)

        for entry in feed.entries:
            torrent = Torrent(entry.title, entry.published_parsed, entry.link)

            if cache.is_new(torrent.published_date, torrent.file_name):
                print(torrent)

                if torrent.save():
                    cache.add(torrent.published_date, torrent.file_name)

        cache.write()
    except Exception as e:
        print(e)
        sys.exit(1)
