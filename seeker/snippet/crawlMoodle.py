#date: 2023-11-01T17:05:02Z
#url: https://api.github.com/gists/070e2ac6148e170dbebd8160b05831d3
#owner: https://api.github.com/users/BillMark98

import requests
from bs4 import BeautifulSoup
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
folderName = "nameOfCourse"
url = "moodle-url"
user_cookie = "here-for-cookie"
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Cookie": user_cookie,
    "Host": "moodle.rwth-aachen.de",
    "Referer": "",
    "Sec-Ch-Ua": '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

response = requests.get(url,headers=headers)
# Find the link (a tag) inside the HTML content

# Extract the link URL and its text
if response.status_code == 200:
    print("Successfully fetched the website content!")
    soup = BeautifulSoup(response.text, 'html.parser')
    link_tags = soup.find_all('a', class_='aalink stretched-link')
    urls_and_names = []
    for link_tag in link_tags:
        link_url = link_tag['href']
        link_text = link_tag.get_text(strip=True)
    # Extract all the option tags
        urls_and_names.append({'url':link_url, 'name':link_text})
    
    # Base URL to complete the relative URLs
    base_url = "https://moodle.rwth-aachen.de"

    iframe_links = []
    results = []
    # Step 4: For each filtered URL, send a request and extract the iframe link.
    for index,item in enumerate(urls_and_names):
        complete_url = item['url']
        response = requests.get(complete_url,headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        pageName = soup.select_one("h1.h2").get_text()
        subfolderName = pageName.strip().replace(" ", "_") + str(index+1)
        path = os.path.join(folderName,subfolderName)
        names = [p.get_text(strip=True) for p in soup.find_all('p',attrs={'dir': 'ltr'}) if len(p.get_text(strip=True))>=1]
        iframes = [iframe for iframe in soup.find_all('iframe', class_='ocplayer')]

        videos = [{"name": n, "iframe_link": i} for n, i in zip(names, iframes)]

        iframe = soup.find('iframe', class_='ocplayer')
        if len(iframes) >= 1:
            for name, iframe in zip(names, iframes):
                results.append({
                    'name': name,
                    'iframe_link': iframe['data-framesrc']
                })
                
        # Save the extracted iframe links.
        print("crawled iframe_links")
        print(iframe_links)

        # convert the ifram_links to m3u8
        
        m3u8_links = [
            f"https://streaming.rwth-aachen.de/rwth/smil:engage-player_{result['iframe_link'].split('/')[-1]}_presentation.smil/playlist.m3u8" 
            for result in results
        ]
        names = [result['name'].replace(" ", "_") for result in results]
        print("m3u8 links:")
        print(m3u8_links)
        print("names:")
        print(names)
        with open("m3u8_links.csv", "w") as f:
            for item in m3u8_links:
                f.write('"%s,"' % item)
            
        with open("names.csv", "w") as f:
            for item in names:
                f.write('"%s,"' % item)

        if not os.path.exists(path):
            os.makedirs(path)
        # use youtube-dl
        # for index, m3u8_link in enumerate(m3u8_links, start=1):
        for name, m3u8_link in zip(names,m3u8_links):
            # strip
            fileName = os.path.join(path, f'{name}.flv')
            cmd = f"youtube-dl {m3u8_link} -o '{fileName}'"
            os.system(cmd)
            print(f"fileName {name} saved")
        
    results = []