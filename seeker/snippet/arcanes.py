#date: 2025-07-08T16:55:22Z
#url: https://api.github.com/gists/0a3617d92dcc0790f02ecf17a25fb7da
#owner: https://api.github.com/users/AnSq

#!/usr/bin/env python

import json
import sys

from requests_ratelimiter import LimiterSession


api_root = "https://api.warframe.market"


def get_prices():
    ses = LimiterSession(per_second=2.8)
    items = ses.get(api_root + "/v2/items", headers={"Language": "en"}).json()["data"]
    arcane_items = list(filter(lambda i: "arcane_enhancement" in i["tags"] and "mod" not in i["tags"], items))
    arcane_items.sort(key=lambda x: x["slug"])
    arcanes = {}
    for i,a in enumerate(arcane_items):
        name = a["i18n"]["en"]["name"]
        url = f"{api_root}/v1/items/{a['slug']}/statistics"
        print(f"[{i}] {name} <{url}>")
        stats = ses.get(url).json()["payload"]["statistics_closed"]["48hours"]

        stats_min = list(filter(lambda x: x["mod_rank"] == 0, stats))
        stats_max = list(filter(lambda x: x["mod_rank"] != 0, stats))

        no_data = {
            "median": None,
            "moving_avg": None,
            "wa_price": None,
            "volume": 0,
            "datetime": ""
        }

        try:
            min_rank = stats_min[-1]
        except IndexError:
            min_rank = dict(no_data)

        try:
            max_rank = stats_max[-1]
        except IndexError:
            max_rank = dict(no_data)

        volume = lambda x: sum(map(lambda i: i["volume"], x))
        min_rank["volume"] = volume(stats_min)
        max_rank["volume"] = volume(stats_max)

        for s in (min_rank, max_rank):
            for k in list(s.keys()):
                if k not in ("median", "moving_avg", "wa_price", "volume", "datetime"):
                    del s[k]
            if "moving_avg" not in s:
                s["moving_avg"] = None

        arcanes[name] = {
            "min_rank": min_rank,
            "max_rank": max_rank,
        }
    with open("arcane_prices.json", "w") as f:
        json.dump(arcanes, f)
    return arcanes


def write_csv(arcanes):
    with open("arcane_prices.csv", "w") as f:
        f.write("Arcane,Med_0,SMA_0,WA_0,Vol48_0,Med_max,SMA_max,WA_max,Vol48_max\n")
        for name, stats in arcanes.items():
            f.write(f"{name},")
            for k in ("min_rank", "max_rank"):
                s = stats[k]
                f.write(f'{s["median"] or ""},{s["moving_avg"] or ""},{s["wa_price"] or ""},{s["volume"]},')
            f.write("\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-x":
        with open("arcane_prices.json") as f:
            arcanes = json.load(f)
    else:
        arcanes = get_prices()
    write_csv(arcanes)

if __name__ == "__main__":
    main()
