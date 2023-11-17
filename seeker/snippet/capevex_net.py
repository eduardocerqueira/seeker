#date: 2023-11-17T16:55:34Z
#url: https://api.github.com/gists/a29be0a3e914875a8762a96a9019a9f9
#owner: https://api.github.com/users/dshaw0004

import os

import pandas as pd
import requests
from bs4 import BeautifulSoup


def capevexnet(html: bytes, site_name: str):
    soup: BeautifulSoup = BeautifulSoup(html, 'html.parser')
    df: pd.DataFrame = pd.DataFrame(
        columns=['title', 'link', 'thumbnail', 'poster', 'video', 'duration'])
    def get_video_link_2(href: str):
        soup: BeautifulSoup = BeautifulSoup(
            requests.get(href).content, 'html.parser')
        video_tag = soup.find('video', attrs= {'class': 'video-js'})
        poster = video_tag['poster']
        video_link = video_tag.source['src']
        return video_link, poster

    if os.path.exists("capevexnet.csv"):
        df = pd.read_csv("capevexnet.csv")

    row_in_df, _ = df.shape

    thumb_blocks = soup.find_all('div', attrs={'class': "th"})

    for index, thumb_block in enumerate(thumb_blocks):

        
        a_tag = thumb_block.find("a")
        href: str = site_name + "/" + a_tag['href']

        img_contianer = a_tag.find('div', attrs={'class': 'th-img'})
        thumbnail: str = site_name + img_contianer.img['src']

        duration = img_contianer.find('span', attrs={'class': 'th-duration'}).text

        title = a_tag.find('div', attrs={'class': 'th-title'}).text

        title = title.replace("\n", '')
        title = title.strip(' ')

        video, poster = get_video_link_2(href)

        df.loc[index + int(row_in_df)] = [
            title, href, thumbnail, site_name + poster, video, duration
        ]

    df.to_csv("capevexnet.csv", index=False)
    
  
if __name__ == "__main__":
    BASE_URL = "https://rapesex.net"
    URL2 = "https://rapesex.net/rapevideo.php"

    res = requests.get(BASE_URL)

    if not res.ok:
        print("did not get any response back\nskipping to next")
    else:
        capevexnet(res.content, BASE_URL)

        print("part - 1 of cornvexnet is done")

    res = requests.get(URL2)

    if not res.ok:
        print("did not get any response back\nend of this function")
        return
    else:
        capevexnet(res.content, BASE_URL)
        print(f"cornvex.tv is done")
