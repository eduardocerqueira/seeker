#date: 2022-03-01T16:57:29Z
#url: https://api.github.com/gists/329621217ba968f5d1e8ea1fecc722d9
#owner: https://api.github.com/users/mkwatson

import json

import requests
from bs4 import BeautifulSoup

url = 'https://www.espn.com/nba/game/_/gameId/401360739'
soup = BeautifulSoup(requests.get(url).text, 'html.parser')
play_by_play = soup.find("div", {"data-plays": True}).attrs['data-plays']

print(json.dumps(json.loads(play_by_play), indent=2))
