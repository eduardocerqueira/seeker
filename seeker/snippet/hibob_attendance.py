#date: 2022-12-20T16:59:13Z
#url: https://api.github.com/gists/8d97ce24e6bf2df15cbf6388817b949e
#owner: https://api.github.com/users/envyj020

#!/usr/bin/env python3

import browser_cookie3
import requests
import datetime
import argparse
import logging
import urllib.parse
from requests.utils import dict_from_cookiejar

logger = logging.getLogger('hibob_attendance')
logger_console = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.addHandler(logger_console)

DATE_FORMAT = "%Y-%m-%d"
TODAY_DATE = datetime.date.today().strftime(DATE_FORMAT)
MORNING = ['08:00', '12:00']
AFTERNOON = ['14:00', '18:00']

parser = argparse.ArgumentParser(description="Hibob attendance script")
parser.add_argument(
        'date', nargs='*', default=[TODAY_DATE],
        help='Date to post the attendace E.g. 2021-02-30. Defaults to today\'s date'
)
parser.add_argument(
        '--browser', '-b', default='chrome', choices=['chrome', 'firefox'],
        help='Browser where to get the cookie from'
)

args = parser.parse_args()

for date in args.date:
    try:
        datetime.datetime.strptime(date, DATE_FORMAT)
    except ValueError:
        logger.error(
            f"Provided date has no valid format ('%Y-%m-%d'): {date}"
        )
        exit(1)

logger.info(f"Getting authentication cookies from {args.browser}")
bc = getattr(browser_cookie3, args.browser)

API_COOKIES, APP_COOKIES = (
        bc(domain_name='app.hibob.com'),
        bc(domain_name='.hibob')
)

EMPLOYEE_ID = urllib.parse \
        .unquote(dict_from_cookiejar(APP_COOKIES).get('ajs_user_id')) \
        .strip('"')

CLOCK_IN_URL = f"https://app.hibob.com/api/attendance/employees/{EMPLOYEE_ID}/attendance/entries"

for date in args.date:
    logger.info(f"Posting attendace for {date}")
    CLOCK_EDIT_URL = f"{CLOCK_IN_URL}?forDate={date}"
    CLOCK_EDIT_DATA = []
    for period in [MORNING, AFTERNOON]:
        CLOCK_EDIT_DATA.append({
            "start": f"{date}T{period[0]}",
            "end": f"{date}T{period[1]}",
            "comment": None, "reason": None, "offset": -60
        })

    response = requests.post(CLOCK_EDIT_URL, verify=True, cookies=API_COOKIES, json=CLOCK_EDIT_DATA)
    response.raise_for_status()
    logger.info(f"Attendace for {date}T{MORNING} - {date}T{AFTERNOON} posted successfully")