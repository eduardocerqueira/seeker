#date: 2025-07-08T16:53:53Z
#url: https://api.github.com/gists/8e249f6c736c708fc7aaf359b698ee3b
#owner: https://api.github.com/users/AnSq

#!/usr/bin/env python3
import json
import math
import os


def main():
    wfinfo_path = os.path.join(os.getenv("APPDATA"), "WFInfo")
    with open(os.path.join(wfinfo_path, "eqmt_data.json")) as f:
        equipment = json.load(f)
    with open(os.path.join(wfinfo_path, "market_data.json")) as f:
        market = json.load(f)

    with open("output.csv", "w") as f:
        f.write('Item,Mastered,Vaulted,Sets,Inventory,Platinum,Ducats,Ducats per Plat,Plat per Ducat,Volume\n')
        for item_name in equipment:
            if item_name == "timestamp":
                continue
            item = equipment[item_name]

            sets = 9999
            for part_name in item["parts"]:
                part = item["parts"][part_name]
                if part_name in equipment:
                    # another prime is used to craft this prime (eg, Bronco -> Akbronco)
                    continue
                sets = min(sets, math.floor(part["owned"] / part["count"]))

            if sets > 0:
                set_item_name = f"{item_name} Set"
                mpart = market[set_item_name]
                plat = mpart["plat"]
                volume = mpart["volume"]
                f.write(f"{set_item_name},{item['mastered']},{item['vaulted']},{sets},{sets},{plat},,,,{volume}\n")

            for part_name in item["parts"]:
                part = item["parts"][part_name]
                if part["owned"] == 0:
                    continue

                try:
                    mpart = market[part_name]
                    plat = mpart["plat"]
                    ducats = mpart["ducats"]
                    volume = mpart["volume"]
                except:
                    plat = None
                    ducats = None
                    volume = 0
                f.write(f"{part_name},{item['mastered']},{item['vaulted']},{sets},{part['owned']},{plat},{ducats},")
                if plat is not None and ducats is not None:
                    f.write(f"{ducats/plat:.2f},{plat/ducats:.2f},")
                else:
                    f.write(",,")
                f.write(f"{volume}\n")


if __name__ == "__main__":
    main()
