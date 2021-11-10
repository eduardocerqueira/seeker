#date: 2021-11-10T16:49:41Z
#url: https://api.github.com/gists/dca341469e1589384c310fe6603e5e9d
#owner: https://api.github.com/users/irfanakyavas

import datetime
import logging
import time
from typing import List, Union
import pandas
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime
from datetime import timedelta
import os
import concurrent.futures
import sys

logger = logging.getLogger("LOG")
logger.setLevel(logging.DEBUG)
SCRIPT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
logger.addHandler(logging.StreamHandler())


def parse_trends_datetime(datetime_string: Union[str, pandas.Timestamp]) -> datetime:
    if isinstance(datetime_string, pandas.Timestamp):
        return parse_trends_datetime_timestamp(datetime_string)
    if isinstance(datetime_string, str) is False:
        datetime_string = pd.to_datetime(str(datetime_string))
        return parse_trends_datetime_timestamp(datetime_string)
    return datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S')


def parse_trends_datetime_timestamp(datetime_string: pandas.Timestamp) -> datetime:
    return datetime.strptime(str(datetime_string), '%Y-%m-%d %H:%M:%S')


class TrendsKeywordTracker:

    def __init__(self, keyword: str, tracking_start_date: datetime, attempt_loading_from_file: bool = True, no_partial: bool = False):
        self.keyword = keyword
        self.tracking_start_date = tracking_start_date
        self.trend_df = pd.DataFrame()
        self.last_value = 1
        self.last_value_date = None
        self.last_value_date: datetime
        self.no_partial = no_partial

        if not attempt_loading_from_file:
            self.download_start_to_today()
        else:
            self.load_from_file()

    def load_from_file(self):
        try:
            self.trend_df = pd.read_csv(f"{self.keyword}.csv", index_col=0)
            logger.log(msg=f"Previous trend data found for {self.keyword}, updating trend data", level=logging.INFO)
            self.update_saved_file()
        except FileNotFoundError as fnfe:
            logger.log(msg=f"No previous trend data found for {self.keyword}\n{fnfe}", level=logging.INFO)

    def save_to_file(self):
        try:
            self.trend_df.to_csv(f"{self.keyword}.csv", mode="w+")
            logger.log(msg=f"Saved trend data for {self.keyword} into {self.keyword}.csv successfully", level=logging.INFO)
        except PermissionError as pe:
            logger.log(msg=f"Error happened while trying to save {self.keyword} trend data as {self.keyword}.csv\n{pe}", level=logging.ERROR)
            logger.log(msg=f"Using alternate saving scheme, trying to save {self.keyword} trend data as {self.keyword}_[timestamp].csv\n"
                           f"You will have to manually rename this csv into {self.keyword}.csv if you want to use it with this script", level=logging.INFO)
            self.trend_df.to_csv(f"{self.keyword}_{datetime.now().timestamp()}.csv", mode="w+")

    def download_start_to_today(self):
        now = datetime.utcnow()
        self.download_trend_between(self.tracking_start_date,
                                    datetime(year=now.year, month=now.month, day=now.day, hour=0, minute=0,
                                             second=0))
        self.save_to_file()
        return self.trend_df

    def update_saved_file(self):
        try:
            self.last_value_date = parse_trends_datetime(self.trend_df.index.values[-1])
            self.last_value = self.trend_df.iloc[-1, 0]
            if self.last_value_date.date() < datetime.utcnow().date():
                now = datetime.utcnow()
                today_second_hour = datetime(year=now.year, month=now.month, day=now.day, hour=1, minute=0,
                                             second=0)
                self.download_trend_between(start=self.last_value_date, end=today_second_hour, last_element=self.last_value)
            else:
                logger.log(msg=f"Data for keyword {self.keyword} is already up to date with current day", level=logging.INFO)
            if self.last_value_date.time().hour < datetime.utcnow().time().hour:
                self.download_today_to_current_hour()
            else:
                logger.log(msg=f"Data for keyword {self.keyword} is already up to date with current hour", level=logging.INFO)
        except IndexError as ie:
            logger.error(
                f"IndexError when reading previous trends data for {self.keyword}, data is corrupt or does not exist. Downloading from beginning.\n{ie}")
            self.download_start_to_today()
            self.update_saved_file()
        self.save_to_file()

    def download_today_to_time(self, spec_time: datetime.time):
        try:
            self.load_from_file()
        except FileNotFoundError as fnfe:
            print(f"No previous trend data found for {self.keyword}, downloading from beginning\n{fnfe}")
            self.download_start_to_today()
            return self.download_today_to_time(spec_time)

        now = datetime.utcnow()
        today_second_hour = datetime(year=now.year, month=now.month, day=now.day, hour=1, minute=0, second=0)
        target_time = datetime(year=now.year, month=now.month, day=now.day, hour=spec_time.hour, minute=spec_time.min,
                               second=spec_time.second)
        self.save_to_file()
        return self.download_trend_between(today_second_hour, target_time)

    def download_today_to_current_hour(self):
        pytrends = TrendReq(tz=360, retries=15)
        print("Downloading to current hour")
        self.last_value_date = parse_trends_datetime(self.trend_df.index.values[-1])
        self.last_value = self.trend_df.iloc[-1, 0]

        now = datetime.utcnow()
        today_current_hour = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour, minute=0,
                                      second=0)
        # print(f"start is {self.last_value_date}")
        # print(f"end is {today_current_hour}")

        new_df = pytrends.get_historical_interest(keywords=[self.keyword],
                                                  year_start=self.last_value_date.year,
                                                  month_start=self.last_value_date.month,
                                                  day_start=self.last_value_date.day,
                                                  hour_start=self.last_value_date.hour,
                                                  year_end=today_current_hour.year,
                                                  month_end=today_current_hour.month,
                                                  day_end=today_current_hour.day,
                                                  hour_end=today_current_hour.hour,
                                                  cat=0, sleep=1)

        # print(new_df.iloc[0, 0])
        adjusting_constant = self.last_value / new_df.iloc[0, 0]
        new_df.iloc[:, 0] = new_df.iloc[:, 0] * adjusting_constant
        self.trend_df = self.trend_df.iloc[:-1, :]
        self.trend_df = self.trend_df.append(new_df)

        series_trend = self.trend_df[self.trend_df.columns[0]]
        max_volume = series_trend.max()
        normalization_factor = 100 / max_volume
        self.trend_df[self.trend_df.columns[0]] = self.trend_df[self.trend_df.columns[0]] * normalization_factor
        print(self.trend_df)

    def download_trend_between(self, start: datetime, end: datetime, last_element: float = -1,
                               max_bucket_size: int = 6):
        last_bucket = False
        pytrends = TrendReq(tz=360, retries=15)

        new_df: pd.DataFrame
        if (end - start).days == 0:
            series_trend = self.trend_df[self.trend_df.columns[0]]
            max_volume = series_trend.max()
            normalization_factor = 100 / max_volume
            self.trend_df[self.trend_df.columns[0]] = self.trend_df[self.trend_df.columns[0]] * normalization_factor
            self.save_to_file()
            print(f"Download of {self.keyword} data done.")
            return self.trend_df
        elif abs((start - end).days) <= max_bucket_size:
            new_df = pytrends.get_historical_interest([self.keyword],
                                                      year_start=start.year,
                                                      month_start=start.month,
                                                      day_start=start.day,
                                                      hour_start=start.hour,
                                                      year_end=end.year,
                                                      month_end=end.month,
                                                      day_end=end.day,
                                                      hour_end=end.hour,
                                                      cat=0, sleep=1)
            last_bucket = True
            bucket_end = end
            print(new_df)
        else:
            bucket_end = start + timedelta(days=max_bucket_size)
            new_df = pytrends.get_historical_interest([self.keyword],
                                                      year_start=start.year,
                                                      month_start=start.month,
                                                      day_start=start.day,
                                                      hour_start=start.hour,
                                                      year_end=bucket_end.year,
                                                      month_end=bucket_end.month,
                                                      day_end=bucket_end.day,
                                                      hour_end=bucket_end.hour,
                                                      cat=0, sleep=1)
            print(new_df)

        print(abs((start - end).days))
        try:
            if last_element != -1:
                adjusting_constant = last_element / new_df.iloc[0, 0]
                new_df.iloc[:, 0] = new_df.iloc[:, 0] * adjusting_constant
        except IndexError:
            print(f"Unknown error b:{start} s:{end} d:{end - start}")
            return self.download_trend_between(start + timedelta(days=1), end, last_element=last_element,
                                               max_bucket_size=max_bucket_size)

        if last_bucket is False:
            new_df = new_df.head(-1)

        new_last_elem = new_df.iloc[-1, 0]
        self.trend_df = self.trend_df.append(new_df)
        time.sleep(0.6)
        return self.download_trend_between(bucket_end, end, last_element=new_last_elem, max_bucket_size=max_bucket_size)


class MultipleTrendsKeywordTracker:

    def __init__(self, keyword_list: List[str], tracking_start_date: datetime, attempt_loading_from_file: bool = False, no_partial: bool = False):
        self.keyword_list = keyword_list
        self.no_partial = no_partial
        self.keyword_trackers: List[TrendsKeywordTracker] = []
        self.tracking_start_date = tracking_start_date
        self.attempt_loading_from_file = attempt_loading_from_file
        [self.init_tracker_for_keyword(keyword) for keyword in self.keyword_list]

    def init_tracker_for_keyword(self, keyword: str):
        self.keyword_trackers.append(TrendsKeywordTracker(keyword, self.tracking_start_date, self.attempt_loading_from_file, self.no_partial))

    def update_trends(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            [executor.submit(trend_tracker.update_saved_file) for trend_tracker in self.keyword_trackers]
        # [trend_tracker.update_saved_file() for trend_tracker in self.keyword_trackers]

    def save_trends_merged(self, file_name: str = ""):
        merge_filename = "_".join(self.keyword_list) if file_name == "" else file_name
        merged_df = pd.concat([trend_tracker.trend_df for trend_tracker in self.keyword_trackers], axis=1)
        merged_df = merged_df.drop(columns=["isPartial"])
        merged_df = merged_df.dropna()
        try:
            merged_df.to_csv(path_or_buf=merge_filename + ".csv", mode="w+")
        except PermissionError as pe:
            logger.log(msg=f"Error when saving merged trends as {file_name}, file is probably in use (in Excel maybe?)\n{pe}", level=logging.ERROR)
            logger.log(msg=f"Using alternate saving scheme, saving merged trends as {file_name}_[timestamp].csv", level=logging.INFO)
            merged_df.to_csv(path_or_buf=f"{merge_filename}_{datetime.datetime.now().timestamp()}.csv", mode="w+")
