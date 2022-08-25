#date: 2022-08-25T17:09:38Z
#url: https://api.github.com/gists/888af25d1054f3619abe3f5ed0d3ce7a
#owner: https://api.github.com/users/real-yfprojects

"""
Calculate the total number of downloads for a flathub app.
"""
import json
import logging
import sys
from datetime import date, timedelta
from typing import Generator, Tuple
import requests

logging.basicConfig(level=logging.INFO, format='%(message)s')

URL_TEMPLATE = 'https://flathub.org/stats/{y:0>4}/{m:0>2}/{d:0>2}.json'


def make_url_list(from_: date, to: date) -> Generator[Tuple[str, date], None, None]:
    td = to - from_
    for day in range(td.days):
        dt = from_ + timedelta(day)
        url = URL_TEMPLATE.format(y=dt.year, m=dt.month, d=dt.day)
        yield url, dt


def download_json(url: str):
    response = requests.get(url)
    response.raise_for_status()

    js = json.loads(response.content)

    return js


def extract_app_stats(json, app_name):
    apps = json['refs']
    if app_name not in apps:
        raise ValueError(f"App {app_name} couldn't be found")

    return apps[app_name]


class DownloadAggregator(Generator[None, dict, Tuple[int, int]]):
    def __init__(self) -> None:
        self.downloads = 0
        self.installs = 0

    def __next__(self) -> None:
        pass

    def __iter__(self):
        return self

    def throw(self, e):
        pass

    def send(self, daily_data: dict) -> None:
        for s in daily_data.values():
            logging.info(s)
            self.downloads += s[0]
            self.installs += s[1]


def stat(app_name: str, last_n_days: int):
    url_list = make_url_list(date.today() - timedelta(days=last_n_days), date.today())
    aggregator = DownloadAggregator()

    for url, dt in url_list:
        logging.info("Processing json for " + str(dt))
        js = extract_app_stats(download_json(url), app_name)
        aggregator.send(js)

    return aggregator.downloads, aggregator.installs


if __name__ == '__main__':
    days = int(sys.argv[1])
    downloads, installs = stat('com.borgbase.Vorta', days)

    print('================================================================')
    print(f"Vorta was downloaded {downloads} times in the past {days} days.")
    print(f"There were {installs} delta downloads.")
