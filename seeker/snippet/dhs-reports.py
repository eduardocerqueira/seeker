#date: 2022-12-23T17:10:02Z
#url: https://api.github.com/gists/cb9096bcc0aa669110aa02dec6cc9cf0
#owner: https://api.github.com/users/testtestingtester

#!/usr/bin/env python
# python dhs-report.py --start-date='2016-09-01'

import os
import sys
import argparse
import pandas as pd
from pandas.tseries.offsets import BDay  # BusinessDay
from datetime import datetime
import requests

class DHS():
    base_url = 'https://www.dhs.gov/sites/default/files/publications/dhs-daily-report-%s.pdf'

    def __init__(self, start_date, end_date):
        if start_date is not None:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date is not None:
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = datetime.today().strftime('%Y-%m-%d')

    def download_reports(self, destination):
        if destination is not None:
            self.destination = destination
        else:
            self.destination = '.'

        if not os.path.isdir(self.destination):
            os.mkdir(self.destination)

        for day in pd.DatetimeIndex(start=self.start_date, end=self.end_date, freq=BDay()):
            self.download_file(self.base_url % day.strftime('%Y-%m-%d'), self.destination)

    def download_file(self, url, destination):
        filename = os.path.join(destination, url.split('/')[-1])
        r = requests.get(url, stream=True)
        if r.ok:
            with open(filename, 'wb') as f:
                print filename
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return filename
        return None
def parse_args():
    parser = argparse.ArgumentParser(
        description='%s - A tool to download DHS Reports' % sys.argv[0],
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", "--destination",  type=str,
                        help="destination where to download all pdf files", default=".")
    parser.add_argument("-s", "--start-date",  type=str,
                        help="start date of reports to download %Y-%m-%d format", required=True)
    parser.add_argument("-e", "--end-date",  type=str,
                            help="end date of reports to download %Y-%m-%d format")
    return parser.parse_args()

def main(*args, **kwargs):
    def use_params(
        destination=None,
        start_date=None,
        end_date=None
    ):
        return DHS(start_date, end_date)\
            .download_reports(destination=destination)

    # argparse arguments
    if len(args) > 0 \
            and isinstance(args[0], argparse.Namespace):
        args = args[0]

        return use_params(
            destination=args.destination,
            start_date=args.start_date,
            end_date=args.end_date
        )
    # dict arguments
    if isinstance(kwargs, dict):
        return use_params(
            destination=kwargs.get('destination'),
            start_date=kwargs.get('start_date'),
            end_date=kwargs.get('end_date')
        )

def use_parse_args():
    args = parse_args()
    return main(args)


def use_keyword_args(**kwargs):
    return main(**kwargs)


if __name__ == '__main__':
    use_parse_args()