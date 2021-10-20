#date: 2021-10-20T17:14:51Z
#url: https://api.github.com/gists/434455c88db64b729e6ecf85aa1aadb6
#owner: https://api.github.com/users/dually8

# python -m pip install beautifulsoup4
# python -m pip install lxml
# python -m pip install requests

import bs4
import requests


def lookUpDeals():
    slickDealsSite = requests.get('https://slickdeals.net/computer-deals/')
    slickDealsSite.raise_for_status()
    soup = bs4.BeautifulSoup(slickDealsSite.text, features="lxml")
    potentialDeals = soup.select('a.itemTitle')
    for i in range(len(potentialDeals)):
        text = potentialDeals[i].string
        isNVME = text.lower().find('nvme') > -1
        is2TB = text.lower().find('2tb') > -1
        if isNVME and is2TB:
            print(f"https://slickdeals.net{potentialDeals[i].get('href')}")

if __name__ == "__main__":
    lookUpDeals()