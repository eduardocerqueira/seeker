#date: 2021-12-02T16:47:33Z
#url: https://api.github.com/gists/5bb3bb55dab0f58f4b249dda92ba49d9
#owner: https://api.github.com/users/ulkayu

# Adapted from example in Ch.3 of "Web Scraping With Python, Second Edition" by Ryan Mitchell

import re
import requests
from bs4 import BeautifulSoup

pages = set()

def get_links(page_url):
  global pages
  pattern = re.compile("^(/)")
  html = requests.get(f"your_URL{page_url}").text # fstrings require Python 3.6+
  soup = BeautifulSoup(html, "html.parser")
  for link in soup.find_all("a", href=pattern):
    if "href" in link.attrs:
      if link.attrs["href"] not in pages:
        new_page = link.attrs["href"]
        print(new_page)
        pages.add(new_page)
        get_links(new_page)
        
get_links("")