#date: 2024-07-10T16:47:52Z
#url: https://api.github.com/gists/b9d9c29f1fd2d07e488cd062a48d4f7d
#owner: https://api.github.com/users/Zhan-Qiu-Liu

import requests
from datetime import datetime, timedelta
import argparse
import time
import random

headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
}


class Hotel:
    def __init__(self, name, brand, country):
        self.name = name
        self.brand = brand
        self.country = country


def get_hotels():
    response = requests.get(
        "https://www.hyatt.com/explore-hotels/service/hotels", headers=headers
    )
    raw_hotels = response.json()
    res = []
    for hotel_key in raw_hotels:
        res.append(
            Hotel(
                raw_hotels[hotel_key]["name"],
                raw_hotels[hotel_key]["brand"]["key"],
                raw_hotels[hotel_key]["location"]["country"]["key"],
            )
        )

    random.shuffle(res)  # Shuffle the list of hotels
    return res


def process_hotel(name):
    url = "https://www.hyatt.com/shop/service/rooms/roomrates/yulzm"

    current_date = datetime.now().date()
    checkinDate = current_date + timedelta(days=30)
    checkoutDate = current_date + timedelta(days=31)

    params = {
        "spiritCode": "yulzm",
        "rooms": "1",
        "adults": "1",
        "location": name,
        "checkinDate": checkinDate.isoformat(),
        "checkoutDate": checkoutDate.isoformat(),
        "kids": "0",
        "rate": "Standard",
    }

    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse response JSON
        if "Credit Card Deposit Required" in response.text:
            print(name, " - Credit Card Deposit Required")
    else:
        print(name, f" - Request failed with status code: {response.status_code}")


def filter_hotels_by_brand_and_country(hotels, brand, country):
    if country:
        return [
            hotel
            for hotel in hotels
            if hotel.brand == brand and hotel.country == country
        ]
    else:
        return [hotel for hotel in hotels if hotel.brand == brand]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter hotels by brand and country")
    parser.add_argument("brand", type=str, help="Brand name to filter by")
    parser.add_argument(
        "--country", type=str, help="Country name to filter by (optional)", default=None
    )
    parser.add_argument("--cookie", type=str, help="Cookie used for Hyatt")
    args = parser.parse_args()
    headers["cookie"] = args.cookie

    hotels = get_hotels()
    filtered_hotels = filter_hotels_by_brand_and_country(
        hotels, args.brand, args.country
    )

    for hotel in filtered_hotels:
        process_hotel(hotel.name)
        delay = random.uniform(5, 15)
        time.sleep(delay)
