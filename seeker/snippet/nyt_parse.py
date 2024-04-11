#date: 2024-04-11T16:57:35Z
#url: https://api.github.com/gists/a67be97955c5760dc1527659922df28e
#owner: https://api.github.com/users/NateM135

import re
from datetime import datetime
from urllib.parse import urlparse, parse_qsl


def validate_nyt_mini_url(url: str) -> bool:
    parsed_url = urlparse(url)

    if not (parsed_url.scheme == "https" and parsed_url.netloc == "www.nytimes.com" and parsed_url.query):
        return False

    params = dict(parse_qsl(parsed_url.query))

    required_params = {"d", "t", "c", "smid"}
    if not required_params.issubset(params):
        return False

    if params["smid"] != "url-share" or not len(params["c"])==32:
        print("here")
        return False

    if not params["t"].isdigit() or int(params["t"]) < 7:
        return False

    if not (len(params["d"]) == 10 and is_valid_date_format(params["d"])):
        return False

    # All validations passed, URL is likely valid
    return True


def is_valid_date_format(date_str: str, format='%Y-%m-%d') -> bool:
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False

def get_date_and_time_strings(url):
    if not validate_nyt_mini_url(url):
        raise ValueError

    params = dict(parse_qsl(urlparse(url).query))
    date_str = params["d"]
    time_str = params["t"]

    # Format date string (assuming YYYY-MM-DD format)
    formatted_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")

    # Time string is already an integer, no formatting needed

    return {"date": formatted_date, "time": time_str}
