#date: 2023-01-03T16:29:19Z
#url: https://api.github.com/gists/52190e20cdb5a8a59bc6fe6cd23295ee
#owner: https://api.github.com/users/BilboTheHobbyist

import json
from pprint import pprint

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

HEADLESS_MODE = True
JSON_FILE = "search-leboncoin.json"
SEARCH_TERM = "Thinkpad"
VERBOSE = True


with sync_playwright() as p:
    browser = p.firefox.launch(headless=HEADLESS_MODE, slow_mo=70)
    page = browser.new_page()
    page.goto(f"https://www.leboncoin.fr/recherche?text={SEARCH_TERM}")

    # Accept cookies
    accept_cookies_button = page.locator("button#didomi-notice-agree-button")
    accept_cookies_button.click()

    # Get HTML source code
    html_source_code = page.content()

    # Parsing HTML
    soup = BeautifulSoup(html_source_code, "html.parser")
    json_content = soup.find("script", {"type":"application/json"}).text
    datas = json.loads(json_content)
    items = datas["props"]["pageProps"]["searchData"]["ads"]

    # Printing datas if VERBOSE
    if VERBOSE:
        for item in items:
            pprint(item)
            print("-" * 100)

    print(f"Écriture des données dans le fichier : {JSON_FILE}")

    with open(JSON_FILE, mode="w") as jsonfile:
        json.dump(items, jsonfile, indent=2)

    browser.close()