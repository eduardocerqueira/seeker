#date: 2025-11-03T16:56:51Z
#url: https://api.github.com/gists/724a5ec6cda501e7cf0fc1600b007a19
#owner: https://api.github.com/users/jlstro

import requests
from bs4 import BeautifulSoup
r = requests.get("https://en.wikipedia.org/wiki/Middle_Rhine", headers={'User-Agent': 'tinyscraper'})
soup = BeautifulSoup(r.text, "html.parser")
for li in soup.find_all("table")[1].find_all("li"):
    text = li.get_text(strip=True)
    link_tag = li.find("a", href=True)
    href = link_tag['href'] if link_tag else None
    print(f"Text: {text} | Link: {href}")