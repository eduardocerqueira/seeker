#date: 2021-11-04T17:14:54Z
#url: https://api.github.com/gists/a9eb07d106d28acde8d7f2cf2f4eb695
#owner: https://api.github.com/users/Dominik1999

# import libraries
import csv
import os.path
import json

from requests_html import HTMLSession
from datetime import datetime
from bs4 import BeautifulSoup

# create an HTML Session object
session = HTMLSession()
# Use the object above to connect to needed webpage
resp = session.get("https://l2fees.info/")
# Run JavaScript code on webpage
resp.html.render()
parsed_html = BeautifulSoup(resp.html.html)
json_data = json.loads(parsed_html.find('script', type='application/json').text)['props']['pageProps']['data']

file_exists = os.path.isfile("l2feesdata.csv")
fieldnames = ["Date",
              "Name",
              "feeTransferEth",
              "feeTransferERC20",
              "feeSwap",
              ]

if not file_exists:
    with open('l2feesdata.csv', 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
        writer.writeheader()

for project in json_data:
    del project["metadata"]
    project["date"] = datetime.now().strftime("%m/%d/%Y")
    data = [project["date"], project["id"], project["results"]['feeTransferEth'], project["results"]['feeTransferERC20'], project["results"]['feeSwap']]
    with open('l2feesdata.csv', 'a') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(data)










