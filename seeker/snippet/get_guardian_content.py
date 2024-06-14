#date: 2024-06-14T16:52:56Z
#url: https://api.github.com/gists/048696c8914d530b30ed7d53e0703464
#owner: https://api.github.com/users/maciejskorski

import os 
import requests
from functools import partial

API_KEY = os.getenv('API_KEY') # NOTE: can use the limited 'test' key for demo purposes
QUERY_TEMPLATE = 'https://content.guardianapis.com/{section}/{year}/{month}/{day}/{title}?api-key={API_KEY}&show-blocks=all' # NOTE: can try less info, e.g. &show-fields=BodyText
QUERY_FN = partial(QUERY_TEMPLATE.format,API_KEY=API_KEY)

web_url = 'https://www.theguardian.com/business/2014/sep/10/thorntons-60-per-cent-profits-rise-despite-closures' # NOTE: put your desired url
section, year, month, day, title = web_url.split('/')[3:]
query = QUERY_FN(section=section,year=year,month=month,day=day,title=title)

results = requests.get(query).json() # content inside 
