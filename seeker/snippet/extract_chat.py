#date: 2024-04-16T16:50:34Z
#url: https://api.github.com/gists/9a57e35e199ce8849cbdfd5a08ae02b9
#owner: https://api.github.com/users/drakedevel

import json
import sys
from bs4 import BeautifulSoup

with open(sys.argv[1]) as in_f:
    soup = BeautifulSoup(in_f.read(), 'html.parser')

data = json.loads(soup.find('script', {'id': '__NEXT_DATA__'}).text)
for entry in data['props']['pageProps']['serverResponse']['data']['linear_conversation']:
    if msg := entry.get('message'):
        role = msg['author']['role']
        if msg['content']['content_type'] != 'text':
            print(f"{role}: [non-text content]")
            continue
        for part in msg['content']['parts']:
            if part:
                print(f"{role}: {part!r}")