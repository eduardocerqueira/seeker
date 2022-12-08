#date: 2022-12-08T17:10:08Z
#url: https://api.github.com/gists/b508664692e71abc510492d284fe462a
#owner: https://api.github.com/users/christopherDT

import pandas as pd
from onboard.client import OnboardClient
try: # for this you can either create a key.py file with one line: api_key = 'your api key here'
    from key import api_key
except ImportError: # or you can just input your api key when you get the prompt
    api_key = input('My dude, kindly enter your api_key')

client = OnboardClient(api_key=api_key)
