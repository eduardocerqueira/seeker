#date: 2023-02-06T17:04:04Z
#url: https://api.github.com/gists/4f5e45d6997a413579877c41b2118a5c
#owner: https://api.github.com/users/5kbpers

#!/bin/python3

import sqlite3
import urllib.parse
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.proxy import Proxy, ProxyType

Beds = 2
Baths = 2
MaxPrice = 4000
MrtStations = [("CC", [20, 24]), ("EW", [15, 28])]

UserAgent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'

BotId = ''
ChannelId = ''
BotToken = "**********"

def build_url():
    url = "https://www.propertyguru.com.sg/property-for-rent?market=residential&listing_type=rent&sort=date&order=desc&"
    params = {
        "beds[]": [Beds],
        "baths[]": [Baths],
        "maxprice": MaxPrice,
        "MRT_STATIONS[]": sum([[l[0]+str(x) for x in range(l[1][0], l[1][1]+1)] for l in MrtStations], []),
    }
    print(sum([[l[0]+str(x) for x in range(l[1][0], l[1][1]+1)] for l in MrtStations], []))
    return url + urllib.parse.urlencode(params, True)

def fetch_properties(url):
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument(f'--user-agent={UserAgent}')
    print(url)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    elems = driver.find_elements(By.CLASS_NAME, 'listing-card')
    properties = list()
    for e in elems:
        info = e.text.split('\n')
        phone = e.find_element(By.CLASS_NAME, 'phone-call-button').get_attribute('href')[4:]
        print(info)
        prop = {
            'id': e.get_attribute('data-listing-id'),
            'name': info[1],
            'price': info[3],
            'available_date': info[4],
            'area': info[6],
            'mrt_distance' : info[8],
            'type': '2b2b ' + info[9],
            'built': info[11],
            'list_date': info[12],
            'link': e.find_element(By.CLASS_NAME, 'nav-link').get_attribute('href'),
            'phone': phone,
            'whatsapp': f"https://api.whatsapp.com/send?phone={phone}",
        }
        properties.append(prop)
    driver.quit()
    return properties

def init_database():
    con = sqlite3.connect("properties.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS property\
        (id, name, price, available_date, area, mrt_distance, type, built, list_date, link, phone, whatsapp)")

def deduplicate_and_save_properties(properties):
    con = sqlite3.connect("properties.db")
    cur = con.cursor()
    res = cur.execute("SELECT id FROM property")
    existed_ids = [x[0] for x in res.fetchall()]
    deduplicated = list()
    for prop in properties:
        if prop['id'] in existed_ids:
            continue
        deduplicated.append(prop)
        print(prop)
        cur.executemany("INSERT INTO property VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(prop['id'], prop['name'], prop['price'], prop['available_date'],
            prop['area'], prop['mrt_distance'], prop['type'], prop['built'],
            prop['list_date'], prop['link'], prop['phone'], prop['whatsapp'],)])
    con.commit()
    return deduplicated

def send_message(properties):
    for prop in properties:
        text = "ID: {}\nName: {}\nPrice: {}\n{}\nArea: {}\n{}\nType: {}\n{}\nList at {} ago\nPhone: {}\nLink: {}\nWhatsapp: {}\n\n".format(
            prop['id'], prop['name'], prop['price'], prop['available_date'],
            prop['area'], prop['mrt_distance'], prop['type'], prop['built'], 
            prop['list_date'], prop['phone'], prop['link'], prop['whatsapp'])
        data = {
            'chat_id': ChannelId,
            'text': text, 
        }
        resp = requests.post(f"https: "**********"
        print(resp.status_code)
        resp.close()

if __name__ == "__main__":
    init_database()
    url = build_url()
    properties = deduplicate_and_save_properties(fetch_properties(url))
    properties.reverse()
    send_message(properties)
se()
    send_message(properties)
