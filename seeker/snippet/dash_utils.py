#date: 2025-03-04T16:58:03Z
#url: https://api.github.com/gists/a5bee20de7749f3a9841ffdefeea6e0a
#owner: https://api.github.com/users/Magnus167

import glob
import os
import pandas as pd
import streamlit
import datetime
import time

CURR_TICKERS_DIR = "../jpmaqs-isc-git/"
PREV_TICKERS_DIR = "../jpmaqs-isc-git - Copy/"


def get_files_list():
    return sorted(glob.glob("data/**/*.csv", recursive=True))


def cache_buster(freq="5min"):
    """
    Returns the current time rounded to the nearest defined time period.
    """
    rounded = pd.Timestamp.now().round(freq)
    return rounded.strftime("%Y%m%d%H%M%S")


@streamlit.cache_data
def _get_report_file(
    prefix: str,
    env: str,
    sep: str,
    date: str = None,
    cache_bust=None,
):
    time.sleep(15)
    env_str = env.strip().upper()
    base_pattern = f"{prefix}{env_str}{sep}"
    files = [
        f for f in get_files_list() if os.path.basename(f).startswith(base_pattern)
    ]
    err_str = f"No report found for prefix={prefix}, env={env}, date={date}"
    if not files:
        raise FileNotFoundError(err_str)
    if date is None:
        return files[-1]
    full_pattern = f"{base_pattern}{date}.csv"
    matching_files = [f for f in files if os.path.basename(f) == full_pattern]
    result = matching_files[0] if len(matching_files) == 1 else None
    if result is None:
        raise FileNotFoundError(err_str)
    return result


def _cached_get_report_file(
    prefix: str,
    env: str,
    sep: str,
    date: str = None,
    cache_bust=None,
):
    cache_bust = cache_bust or cache_buster()
    return _get_report_file(prefix, env, sep, date, cache_bust)


def get_calc_error_report(env="PROD", date=None):
    file_path = _cached_get_report_file("calc-error_", env, "_", date)
    return pd.read_csv(file_path)


def get_collect_error_report(env="PROD", date=None):
    file_path = _cached_get_report_file("Collect-Errors-", env, "-", date)
    return pd.read_csv(file_path)


def get_discontinued_series_report(env="PROD", date=None):
    file_path = _cached_get_report_file("Discontinued-Series-", env, "-", date)
    return pd.read_csv(file_path)


@streamlit.cache_data
def _get_tickers_list(cache_bust=None):
    curr_files = glob.glob(f"{CURR_TICKERS_DIR}**/*.csv", recursive=True)
    prev_files = glob.glob(f"{PREV_TICKERS_DIR}**/*.csv", recursive=True)
    curr_files = list(map(os.path.basename, curr_files))
    prev_files = list(map(os.path.basename, prev_files))
    curr_files = sorted(set(curr_files + prev_files))
    curr_files = [f.replace(".csv", "") for f in curr_files]
    return curr_files


def get_tickers_list():
    return _get_tickers_list(cache_bust=cache_buster("1min"))


@streamlit.cache_data
def _get_ticker_files(
    ticker: str,
    curr_date: str = None,
    prev_date: str = None,
    cache_bust=None,
):
    def get_ticker_file(ticker, folder):
        files = glob.glob(f"{folder}**/{ticker}.csv", recursive=True)
        if not files:
            raise FileNotFoundError(f"No file found for ticker={ticker}")
        return pd.read_csv(files[0])

    curr_file = get_ticker_file(ticker, CURR_TICKERS_DIR)
    prev_file = get_ticker_file(ticker, PREV_TICKERS_DIR)
    return curr_file, prev_file


def _cached_get_ticker_files(
    ticker: str,
    curr_date: str = None,
    prev_date: str = None,
    cache_bust=None,
):
    cache_bust = cache_bust or cache_buster()
    return _get_ticker_files(ticker, curr_date, prev_date, cache_bust)


def get_ticker_files(ticker: str, curr_date: str = None, prev_date: str = None):
    return _cached_get_ticker_files(ticker, curr_date, prev_date)