#date: 2022-01-18T17:10:18Z
#url: https://api.github.com/gists/53dd1022446fa33fbf6345df3d04a4b7
#owner: https://api.github.com/users/perymerdeka

import requests
from bs4 import Beautifulsoup

url = 'https://www.yell.com/ucs/UcsSearchAction.do?'

params = {
  'scrambleSeed': '383401705',
  'keywords': 'hotel',
  'location': 'New York'
}

headers = {

  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36'
}


res = requests(url, params=params, headers=headers)
soup = Beautifulsoup(res.text, 'html.parser')


# scraper disini



