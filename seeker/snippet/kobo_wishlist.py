#date: 2025-08-25T17:03:18Z
#url: https://api.github.com/gists/32162d1e9423a04c02b1d3d0bb455a79
#owner: https://api.github.com/users/thiagomgd

from pathlib import Path
from urllib.request import urlopen
from pprint import pprint
from tqdm import tqdm
from time import sleep
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import json
import re 

options = ChromeOptions()
# options.add_argument("--headless")
browser = webdriver.Chrome(options=options)
# browser = webdriver.Chrome(executable_path="/opt/homebrew/bin/chromedriver", options=options)

MAX_POINTS = 6400
SALE_PRICE = 6
PERCENT_DISCOUNT = 30

WISHLIST = [
    "https://www.kobo.com/ca/en/ebook/the-bittlemores",
    "https://www.kobo.com/ca/en/ebook/sackett-s-land",
    "https://www.kobo.com/ca/en/ebook/the-son-2",
    "https://www.kobo.com/ca/en/ebook/the-last-unicorn-9",
    "https://www.kobo.com/ca/en/ebook/boy-s-life",
    "https://www.kobo.com/ca/en/ebook/lonesome-dove-2",
    "https://www.kobo.com/ca/en/ebook/something-wicked-this-way-comes"
]

def getPrice(text):
    num = ""
    for c in text:
        if c.isdigit() or c == '.':
            num = num + c

    if num == '':
        return 0

    return float(num)

def checkExists(browser, element):
    try: 
        browser.find_element(By.CSS_SELECTOR, (element))
        return True
    except:
        return False

def isOkSale(book):
    if book["dealDescription"]:
        return True
    
    if book["isSale"] == False:
        return False
    
    if book["price"] <= SALE_PRICE:
        return True

    if book["discountPercent"] >= PERCENT_DISCOUNT:
        return True
    
    return False

def getBookData(url):
    # print(url)
    while True:
        try:
            browser.get(url)
            break
        except Exception as e:
            # print(f"Retrying due to error: {e}")
            sleep(2)

    WebDriverWait(browser, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.active-price > div.price-wrapper > span.price:not(:empty)"))
            )

    title = browser.find_element(By.CSS_SELECTOR, ("h1.title")).get_attribute('textContent').strip()
    priceText = browser.find_element(By.CSS_SELECTOR, ("div.active-price > div.price-wrapper")).get_attribute('textContent')
    cover = browser.find_element(By.CSS_SELECTOR, ("img.cover-image")).get_attribute("src")
    author = browser.find_element(By.CSS_SELECTOR, ("a.contributor-name")).get_attribute("textContent")
    

    preorder = ''

    if checkExists(browser, 'p.preorder-subtitle'):
        preorder = browser.find_element(By.CSS_SELECTOR, "p.preorder-subtitle").get_attribute("textContent")

    isPlus = checkExists(browser, 'h2.subscription-title')
    isSale = checkExists(browser, 'span.saving-callout') or checkExists(browser, "div.original-price")
    discountText = browser.find_element(By.CSS_SELECTOR, 'span.saving-callout').get_attribute('textContent') if checkExists(browser, 'span.saving-callout') == True else ""
    discountPercent = 0 if discountText == "" else int(re.findall(r'\d+', discountText)[0])
    dealDescription = browser.find_element(By.CSS_SELECTOR, "div.deal-description").get_attribute("textContent") if checkExists(browser, 'div.deal-description') else ''
    price = getPrice(priceText)

    pointsText = browser.find_element(By.CSS_SELECTOR, 'div.pricing-footer > span').get_attribute('textContent') if checkExists(browser, 'div.pricing-footer > span') == True else ""
    points = 0 if pointsText == "" else int(re.findall(r'\d+', pointsText)[0])

    return {
        'title': title,
        'price': price,
        'cover': cover,
        'url': url,
        'preorder': preorder,
        'isPlus': isPlus,
        'isSale': isSale,
        'discountPercent': discountPercent,
        'author': author,
        'dealDescription': dealDescription,
        'points': points
    }

books = []

for url in tqdm(WISHLIST):
    sleep(0.5)
    book = getBookData(url)

    if (book['price'] == 0):
        # input('waiting')
        sleep(2)
        book = getBookData(url)

    if (book['price'] == 0):
        sleep(2)
        book = getBookData(url)

    books.append(book)


onSale = [{"title": x["title"], "price": x["price"], "url": x["url"], "dealDescription":x["dealDescription"]} for x in books if isOkSale(x)]
onSale.sort(key=lambda x: (x['price'], x['title']), reverse=False)

koboPlus = [x["url"] for x in books if x["isPlus"] == True]

booksByPointsDict = {}
booksByPricePointBalanceDict = {}
for book in books:
    points = book['points']

    if points > MAX_POINTS or points == 0:
        continue
    if points not in booksByPointsDict:
        booksByPointsDict[points] = []
    booksByPointsDict[points].append({"title": book["title"], "price": book["price"], "url": book["url"]})

    # round price so 9.95 and 9.99 have the same point per dollar balance
    pricePoint = round(book["points"] / round(book["price"]))

    if pricePoint not in booksByPricePointBalanceDict:
        booksByPricePointBalanceDict[pricePoint] = []
    booksByPricePointBalanceDict[pricePoint].append({"title": book["title"], "price": book["price"], "url": book["url"]})

booksByPoints = []
for points, books in booksByPointsDict.items():
    books.sort(key=lambda x: x['title'], reverse=False)
    booksByPoints.append({"points": points, "books": books})
booksByPoints.sort(key=lambda x: x['points'], reverse=False)


booksByPricePoint = []
for pricePoint, books in booksByPricePointBalanceDict.items():
    books.sort(key=lambda x: x['title'], reverse=False)
    booksByPricePoint.append({"pricePoint": pricePoint, "books": books})
booksByPricePoint.sort(key=lambda x: x['pricePoint'], reverse=True)

bookInfo = {"sale": onSale, "plus": koboPlus, "points": booksByPoints, "priceAndPointBalance": booksByPricePoint}

with open('bookInfo.json', "w") as f:
    json.dump(bookInfo, f, indent=4)

browser.quit()