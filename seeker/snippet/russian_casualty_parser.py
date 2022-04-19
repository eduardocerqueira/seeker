#date: 2022-04-19T17:13:00Z
#url: https://api.github.com/gists/5197512061ed84fa6082c09852085142
#owner: https://api.github.com/users/YuriyGuts

# -*- coding: utf-8 -*-
import datetime
import re

from bs4 import BeautifulSoup
import pandas as pd
import requests


CATEGORY_NORMALIZATION_MAP = {
    "РСЗВ Град": "РСЗВ",
    "ЗРК БУК": "Засоби ППО",
}


def download_monthly_data(month_string):
    url = f"https://index.minfin.com.ua/ua/russian-invading/casualties/month.php?month={month_string}"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_daily_data(day_element):
    date = day_element.find("span", {"class": "black"}).text
    date = datetime.datetime.strptime(date, "%d.%m.%Y").strftime("%Y-%m-%d")

    result = {"Дата": date}

    categories = day_element.find("div", {"class": "casualties"}).find_all("li")
    for category_element in categories:
        title, count = category_element.text.split("—")[:2]
        title = title.strip()
        title = CATEGORY_NORMALIZATION_MAP.get(title, title)

        count = re.findall(r"\d+", count)[0]
        result[title] = int(count)

    return result


def parse_monthly_data(month_string):
    downloaded_data = download_monthly_data(month_string)
    soup = BeautifulSoup(downloaded_data, "html.parser")

    result = []
    for day_element in soup.find_all("li", {"class": "gold"}):
        result.append(parse_daily_data(day_element))

    return result


def main():
    all_data = []
    all_data.extend(parse_monthly_data("2022-04"))
    all_data.extend(parse_monthly_data("2022-03"))
    all_data.extend(parse_monthly_data("2022-02"))

    df_absolute = pd.DataFrame.from_records(all_data).set_index("Дата").sort_values(by="Дата")
    df_delta = df_absolute.diff(periods=1, axis=0)

    df_absolute.to_csv("casualties_absolute.csv", header=True)
    df_delta.to_csv("casualties_delta.csv", header=True)


if __name__ == "__main__":
    main()
