#date: 2023-07-14T16:59:37Z
#url: https://api.github.com/gists/3b43c030c9dfbd702949cdfb01027bd6
#owner: https://api.github.com/users/imaspacecat

import scrapy
from dotenv import dotenv_values
import requests as req
from scrapy.http import FormRequest
from pathlib import Path
import sys


current_dir = Path(__file__)
PROJECT_NAME = 'sim-companies-api'
ROOT_DIR = next(
    p for p in current_dir.parents if p.parts[-1] == PROJECT_NAME
)
print("root dir:", ROOT_DIR)
print(type(ROOT_DIR))
sys.path.insert(1, str(ROOT_DIR) + "/src/")
import core

env_data = dotenv_values(str(ROOT_DIR)+"/.env")
email = env_data["EMAIL"]
password = "**********"


simco = "**********"
simco.login()
print("cookies:", simco.get_cookies())



class APISpider(scrapy.Spider):
    name = "api"

    def __init__(self):
        self.links=[]

    # def parse(self, response):
    #     self.links.append(response.url)
    #     for href in response.css('a::attr(href)'):
    #         yield response.follow(href, self.parse)

    def start_requests(self):
        urls = [
            "https://www.simcompanies.com/landscape",
        ]

        for url in urls:
            yield scrapy.Request(url=url, cookies=simco.get_cookies(), callback=self.parse)

    def parse(self, response):
        print("response url:", response.url)

        self.links.append(response.url)
        print(self.links)
        
        page = response.url.split("/")[-2]
        filename = f"quotes-{page}.html"
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")

        for href in response.css('a::attr(href)'):
            yield response.follow(href, self.parse)
llow(href, self.parse)
