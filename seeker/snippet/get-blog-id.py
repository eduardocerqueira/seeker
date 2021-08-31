#date: 2021-08-31T01:22:56Z
#url: https://api.github.com/gists/a0919fdbb3048395f0bf63f16250fd75
#owner: https://api.github.com/users/RellikJaeger

# Description: Fetching Google blogger blog id using Python3 script.
#            : You need to run with Python3 and you may need to install some modules as below.
#            : If you have multiple pips, do: python -m pip install requests beautifulsoup4.
#            : Else, do: pip install requests beautifulsoup4.

# Usage      : python get-blog-id.py https://yourblog.blogspot.com

import sys
import requests
from bs4 import BeautifulSoup as soup

def main():
    try:
        url = sys.argv[1]
        print('\n URL:', url)
        response = requests.get(url)
        html = response.text
        doc = soup(html, 'html.parser')
        for link in doc.findAll('link', {'rel':'service.post'}):
            list = link.get('href').split('/')
            id = list[len(list) - 3]
            print('=============================================\n Google Blogger Blog ID:', id, '\n=============================================')
    except Exception as e:
        if hasattr(e, 'message'):
            print('\n', e.message)
        else:
            print('\n', e)

if __name__ == "__main__": main()
