#date: 2022-07-18T17:03:38Z
#url: https://api.github.com/gists/91876f97747413a738b715482e7b9036
#owner: https://api.github.com/users/bbelderbos

from collections import Counter
from datetime import date

from dateutil.parser import parse
import plotext as plt
import requests

API_URL = "https://codechalleng.es/api/articles/"
START_YEAR = 2017
THIS_YEAR = date.today().year
THIS_MONTH = date.today().month
MONTH_RANGE = range(1, 13)


def _create_yymm_range():
    for year in range(START_YEAR, THIS_YEAR + 1):
        for month in MONTH_RANGE:
            yield f"{year}-{str(month).zfill(2)}"
            if year == THIS_YEAR and month == THIS_MONTH:
                break


def get_articles_per_month(url=API_URL):
    ym_range = _create_yymm_range()
    cnt = Counter({ym: 0 for ym in ym_range})
    data = requests.get(API_URL)
    for row in data.json():
        dt = parse(row["publish_date"])
        if dt.year < START_YEAR:
            continue
        ym = dt.strftime("%Y-%m")
        cnt[ym] += 1
    return cnt


def show_plot(data):
    labels, values = zip(*data.items())
    plt.bar(labels, values)
    plt.title("Pybites articles published per month")
    plt.show()


if __name__ == "__main__":
    data = get_articles_per_month()
    show_plot(data)
