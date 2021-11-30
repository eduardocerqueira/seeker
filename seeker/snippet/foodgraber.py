#date: 2021-11-30T17:09:36Z
#url: https://api.github.com/gists/d4ee8c323426e8ad237027494e6112f8
#owner: https://api.github.com/users/BlueDev1

import requests
from bs4 import BeautifulSoup
from lxml import etree
url = requests.get("http://www.os-brace-radica-klostarivanic.skole.hr/_kolska_kuhinja")
soup = BeautifulSoup(url.content, 'html.parser')
dom = etree.HTML(str(soup))
food = []
date = []


def hrana(dom,food):
    food.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[1]/td[2]/p/font/font/font/span')[0].text)
    food.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[2]/td[2]/p/font/font/font/span')[0].text)
    food.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[3]/td[2]/p/font/font/font/span')[0].text)
    food.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[4]/td[2]/p/font/font/font/span')[0].text)
    food.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[5]/td[2]/p/font/font/font/span')[0].text)

hrana(dom,food)


def datum(date):
    date.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[1]/td[1]/p/font/font/font/span')[0].text)
    date.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[2]/td[1]/p/font/font/font/span')[0].text)
    date.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[3]/td[1]/p/font/font/font/span')[0].text)
    date.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[4]/td[1]/p/font/font/font/span')[0].text)
    date.append(dom.xpath('//*[@id="glavni-stupac"]/div[2]/div[2]/div[3]/div[2]/table/tbody/tr[5]/td[1]/p/font/font/font/span')[0].text)

datum(date)



def combine(food,date):
    for i in range(0, len(food)):    
        for i in range(0, len(date)):
            result = date[i] + food[i]
            print(result)

combine(food,date)