#date: 2023-01-23T16:55:35Z
#url: https://api.github.com/gists/cd74bbac9d5208b738ba4209d4855241
#owner: https://api.github.com/users/elektr0nisch

import csv

from datetime import datetime
from pprint import pprint

data = []

with open("data.csv", newline="", encoding="utf8") as file:
    reader = csv.DictReader(file, delimiter=";")
    for row in reader:
        data.append(row)

data = sorted(data, key=lambda row: int(row["Vorgangsnummer"]))

totalMoney = 0
personMoneyMap = {}
lastRow = None
for row in data:
    date = datetime.strptime(row["Datum"], "%d.%m.%Y %H:%M:%S")

    if lastRow:
        if row["Loginname"] == lastRow["Loginname"] and row["Ware"] == lastRow["Ware"]:
            lastDate = datetime.strptime(lastRow["Datum"], "%d.%m.%Y %H:%M:%S")
            delta = date - lastDate

            if delta.total_seconds() <= 30:
                if row["Loginname"] in personMoneyMap:
                    data = personMoneyMap[row["Loginname"]]
                else:
                    data = {"money": 0, "count": 0}

                price = float(row["Preis"].strip("€").replace(",", "."))

                totalMoney += price

                data["money"] += price
                data["count"] += 1
                personMoneyMap[row["Loginname"]] = data

    lastRow = row

with open("duplicates.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file, delimiter=";")

    writer.writerow(["Benutzername", "Anzahl Doppelbuchungen", "Rückerstattungsbetrag"])

    for username, data in personMoneyMap.items():
        writer.writerow(
            [username, data["count"], str(data["money"]).replace(".", ",") + " €"]
        )