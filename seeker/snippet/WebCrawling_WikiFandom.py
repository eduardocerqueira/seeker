#date: 2023-04-27T17:08:56Z
#url: https://api.github.com/gists/35771a7d41f804e02f58857ccb5ebe37
#owner: https://api.github.com/users/BOAScripts

#!/usr/bin/python3 

'''
Downloading/Updating Color spreads of One Piece from the fandom site.

Usage:
python ./WebCrawling_wikiFandom.py 
    --> will download all the images in the current directory

python ./WebCrawling_wikiFandom.py -p "{PATH-TO-UPDATE}"
    --> will download in the provided directory the latest color spreads. (can also be used to download all spreads to a specific directory)
    --> do not end the path with a "/", it may break the script. to lazy to handle error...
'''

import requests
from bs4 import BeautifulSoup
import os
import argparse

# URL where there is a list of all the imgs to download
baseUrl = "https://onepiece.fandom.com/wiki/Category:Color_Spreads"

## Cleanup output
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# PARAMETER verification
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pathToUpdate", type=str, required=False)
args = parser.parse_args()

if args.pathToUpdate:
    imgPath = args.pathToUpdate
    localImgList = []
    for item in os.listdir(imgPath):
        if (os.path.isfile(f"{imgPath}/{item}")) and ((item.endswith(".png")) or (item.endswith(".jpg")) or (item.endswith(".jpeg"))):
            localImgList.append(item)
    print(f"Updating {imgPath} --> number of local images : {len(localImgList)}")
    update = True
else:
    imgPath = (os.getcwd()).replace("\\", "/")
    print(f"Downloading all images in current folder: {imgPath}")
    update = False

# DEFINITION
def DownloadImg(session, url,name,path):
    '''
    Download the content of a get request.
    '''
    r = session.get(url)
    if r.status_code == 200:
        with open(path + "/" + name, 'wb') as f:
            f.write(r.content)
    else:
        (f"Failed to get: {url} -- Status Code: {r2.status_code}")

# Session
s = requests.Session()
# Base GET request
r = s.get(baseUrl)
if r.status_code == 200:
    print(f"{baseUrl}: 200 - OK")
    # Parse Base HTML
    soup = BeautifulSoup(r.text, 'html.parser')
    lis = (soup.find('ul',{"class": "category-page__members-for-char"})).find_all('li')
    # Populate a list of remote image names
    remoteImgList = []
    for li in lis:
        imgSrc = li.find('img')['src']
        imgUrl = imgSrc.split("/revision/")[0]
        imgName = imgUrl.split("/")[-1]
        remoteImgList.append(imgName)
    
    print(f"Total number of remote images: {len(remoteImgList)}")
    # Download images following the passed parameter.
    if update:
        if sorted(remoteImgList) == sorted(localImgList):
            print(f"{imgPath} up-to-date. No new images to download")
        else:
            counter = 0
            for li in lis:
                imgSrc = li.find('img')['src']
                imgUrl = imgSrc.split("/revision/")[0]
                imgName = imgUrl.split("/")[-1]
                
                if imgName not in localImgList:
                    print(f"Downloading: {imgName} in {imgPath}")
                    print(LINE_UP, end=LINE_CLEAR)
                    DownloadImg(s, imgUrl, imgName, imgPath)
                    counter += 1

            print(f"Downloaded {counter} images in {imgPath}")
    else:
        counter=0
        for li in lis:
            imgSrc = li.find('img')['src']
            imgUrl = imgSrc.split("/revision/")[0]
            imgName = imgUrl.split("/")[-1]

            print(f"Downloading: {imgName}")
            print(LINE_UP, end=LINE_CLEAR)
            DownloadImg(s, imgUrl, imgName, imgPath)
            counter += 1
        print(f"Downloaded {counter} images in {imgPath}")
else:
    print(f"{baseUrl}: {r.status_code} - NOT OK")

