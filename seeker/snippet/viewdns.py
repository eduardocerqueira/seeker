#date: 2023-08-02T17:02:34Z
#url: https://api.github.com/gists/200e09908ebd4edf41be2f4208c1ccd5
#owner: https://api.github.com/users/null3yte

import urllib.parse as encode
import requests
from bs4 import BeautifulSoup

def viewDNS(company: str) -> set:
    domains = set()
    
    query_encode: str = encode.quote(company)
    search_url: str = f"https://viewdns.info/reversewhois/?q={query_encode}"
    
    headers: dict = {
        'Host': 'viewdns.info',
        'Sec-Ch-Ua': '"Not:A-Brand";v="99", "Chromium";v="112"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.50 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    response = requests.get(search_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'border': '1'})
        if table:
            rows = table.find_all('tr')
            for row in rows[1:]:
                columns = row.find_all('td')
                domain = columns[0].text.strip()
                domains.add(domain)
    return domains