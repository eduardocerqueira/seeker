#date: 2023-12-11T16:53:41Z
#url: https://api.github.com/gists/547db1095dce8f8da9803db88a75997a
#owner: https://api.github.com/users/erreurBarbare

from bs4 import BeautifulSoup
import time
import random
import requests
from collections import Counter

URL_KAFKA_JOBS = 'https://www.jobscout24.ch/de/jobs/kafka/?catid=8'
USER_AGENT = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                            'Chrome/110.0.0.0 Safari/537.36'}

def get_request_headers(url):
    headers = USER_AGENT
    return headers.update({'referer': url})


def get_page(url):
    resp = requests.get(url, headers=get_request_headers(f"{URL_KAFKA_JOBS}"))
    if resp.status_code == 200:
        return resp
    else:
        return None

page_id = 1
download_ok = True

companies = []

while download_ok:
    download = get_page(f"{URL_KAFKA_JOBS}&p={page_id}")

    if download is None:
        print(f"found a total of {page_id - 1} result pages.")
        break

    page_i = BeautifulSoup(download.content, 'html.parser')
    jobs = page_i.find_all('li', {"class": "job-list-item"})
    for job in jobs:
        company = job.find('p', {'class': 'job-attributes'}).find('span').text.strip()
        companies.append(company)

    page_id += 1
    time.sleep(random.randint(1, 3))

COMPANIES_SORTED = (Counter(companies).most_common())

for c in COMPANIES_SORTED:
    print(f"{c[0]}, {c[1]}")