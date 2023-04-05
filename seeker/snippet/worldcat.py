#date: 2023-04-05T16:59:01Z
#url: https://api.github.com/gists/760df1f82a2555d34d7deb01c8f3c39a
#owner: https://api.github.com/users/johannahrodgers

from bs4 import BeautifulSoup
import requests
import urllib.parse
import pandas as pd
import json
import csv


def build_query(a, b):
    a = a.replace(' ', '+')
    b = urllib.parse.quote(b)
    return f'https://www.worldcat.org/search?q={a}%20{b}'


def lovely_soup(url):
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'})
    return BeautifulSoup(r.text, 'lxml')


def get_first_edition_date(title, author):
    url = build_query(title, author)
    soup = lovely_soup(url)
    editions_urls = soup.find_all('a', title='View all held editions and formats for this item')
    for editions_url in editions_urls:
        editions_url = editions_url['href']
        editions_url = f'https://www.worldcat.org{editions_url}&qt=sort_yr_asc&sd=asc&start_edition=1'
        soup = lovely_soup(editions_url)
        table = soup.find('table', {'class': 'table-results'})
        df = pd.read_html(str(table))
        data = json.loads(df[0].to_json(orient='records'))
        for row in data:
            if row['Date / Edition']:
                return int(row['Date / Edition'])


with open('data.csv', newline='') as csvfile:
    csv_data = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in csv_data:
        title = row[1]
        author = row[0].split(';')[0].split(',')[0]
        if title != 'Title':
            first_edition_date = get_first_edition_date(title, author)
            print(f'Title: {title}\nAuthor: {author}\nDate: {first_edition_date}\n')
