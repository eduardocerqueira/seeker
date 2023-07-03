#date: 2023-07-03T17:04:33Z
#url: https://api.github.com/gists/7924525a48df8cdd07804a78a9e39f9b
#owner: https://api.github.com/users/arnu515

from bs4 import BeautifulSoup
from emojiflags.lookup import lookup
import requests
import csv


def scrape_iso3166_country_codes(url):
  response = requests.get(url)
  if response.status_code != 200:
    raise Exception("Failed to fetch the page content.")

  soup = BeautifulSoup(response.content, 'html.parser')
  table = soup.find('table', {'class': 'wikitable'})

  if not table:
    raise Exception("Table not found in the page.")

  rows = table.find_all('tr')

  data = []
  for row in rows[1:]:  # Skip the header row
    columns = row.find_all('td')
    if len(columns) >= 4:  # Check if it's a valid row with at least 4 columns
      country_name = columns[1].text.strip()
      flag_img = columns[0].find('img')
      flag_img_url = flag_img['src'] if flag_img else ''
      alpha2_code = columns[3].text.strip()
      alpha3_code = columns[4].text.strip()

      data.append([
        country_name, "https:" + flag_img_url, alpha2_code, alpha3_code,
        lookup(alpha2_code)
      ])

  return data


if __name__ == "__main__":
  url = "https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes"
  iso3166_country_codes = scrape_iso3166_country_codes(url)
  csv_filename = "countries.csv"
  with open(csv_filename, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['name', 'flag_url', '2letter', '3letter', 'emoji'])
    writer.writerows(iso3166_country_codes)
