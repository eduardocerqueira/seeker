#date: 2025-06-25T16:51:42Z
#url: https://api.github.com/gists/ad6165e885e691187210a9f5484307a4
#owner: https://api.github.com/users/qiaant

# Original script by drdaxxy
# https://gist.github.com/drdaxxy/1e43b3aee3e08a5898f61a45b96e4cb4

# Thanks to ToshiroScan for fixing after a shonen update broke this.
# And QPDEH for helping fix the black bars issue with some manga.
# This script has been tested to work with python 3.10

# To install required libraries: 
#
# pip install requests
# pip install Pillow
# pip install beautifulsoup4

import sys
import os
import requests
import errno
import json
from PIL import Image
from bs4 import BeautifulSoup

login = False # Set this to True if you want to login
username = "your email here"
password = "**********"


loginheaders = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0',
    'origin': 'https://pocket.shonenmagazine.com',
    'x-requested-with': 'XMLHttpRequest'
}

loginurl = "https://pocket.shonenmagazine.com/user_account/login"

sess = requests.Session()

if login:

    logindata = {"email_address": "**********": password, "return_location_path" : "/"}

    r = sess.post(loginurl, headers=loginheaders, data = logindata)

    if r.ok:

        print('LOGIN SUCCESS')
        print(sess.cookies)
    else:
        print('LOGIN FAILED')
        print(r.headers)
        print(r.status_code)
        print(r.reason)
        print(r.text)



if len(sys.argv) != 3:
    print("usage: shonenripperjson.py <url> <destination folder>")
    sys.exit(1)


destination = sys.argv[2]

if not os.path.exists(destination):
    try:
        os.makedirs(destination)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

url = sys.argv[1]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0'}

# If soup method below doesn't work, try uncommenting these 2 lines

# if not url.endswith('.json'):
#     url = url + ".json"

print("Getting from url: "+url)
r = sess.get(url=url, headers=headers)

# And this
# data = r.json()

# and comment from here ========

soup = BeautifulSoup(r.content, 'html.parser')

script_tag = soup.find('script', id='episode-json')

if script_tag:
    json_data = script_tag['data-value']
    
    data = json.loads(json_data)
else:
    print("No <script> with ID 'episode-json' found.")
    sys.exit(1)

# to here ======================


def dlImage(url, outFilename, drm):
    r = sess.get(url, stream=True, headers=headers)

    if not r.ok:
        print(r)
        return

    content_type = r.headers.get('content-type')
    if content_type == "image/jpeg":
        outFilename = outFilename + ".jpg"
    elif content_type == "image/png":
        outFilename = outFilename + ".png"
    else:
        print("content type not recognized!")
        print(r)
        return

    with open(outFilename, 'wb') as file:
        for block in r.iter_content(1024):
            if not block:
                break
            file.write(block)

    if drm == True:
        source = Image.open(outFilename)
        dest = source.copy()
        
        def draw_subimage(sx, sy, sWidth, sHeight, dx, dy):
            rect = source.crop((sx, sy, sx+sWidth, sy+sHeight))
            dest.paste(rect, (dx, dy, dx+sWidth, dy+sHeight))

        DIVIDE_NUM = 4
        MULTIPLE = 8
        cell_width = (source.width // (DIVIDE_NUM * MULTIPLE)) * MULTIPLE
        cell_height = (source.height // (DIVIDE_NUM * MULTIPLE)) * MULTIPLE
        for e in range(0, DIVIDE_NUM * DIVIDE_NUM):
            t = e // DIVIDE_NUM * cell_height
            n = e % DIVIDE_NUM * cell_width
            r = e // DIVIDE_NUM
            i_ = e % DIVIDE_NUM
            u = i_ * DIVIDE_NUM + r
            s = u % DIVIDE_NUM * cell_width
            c = (u // DIVIDE_NUM) * cell_height
            draw_subimage(n, t, cell_width, cell_height, s, c)

        dest.save(outFilename)


if 'readableProduct' in data:
    readableProduct = data['readableProduct']
    nextReadableProductUri = None

    if 'nextReadableProductUri' in readableProduct:
        nextReadableProductUri = readableProduct['nextReadableProductUri']

    if 'pageStructure' in readableProduct:
        pageStructure = readableProduct['pageStructure']

        if pageStructure == None:
            print('Could not download pages. Most likely this volume is not public.')
            sys.exit(1)

        choJuGiga = pageStructure['choJuGiga'] if 'choJuGiga' in pageStructure else ''

        print('choJuGiga: ', choJuGiga)

        drm = choJuGiga != "usagi"

        pages = pageStructure['pages'] if 'pages' in pageStructure else []

        if len(pages) == 0:
            print("No pages found")
            sys.exit(1)

        pageIndex = 0

        for page in pages:
            if 'src' in page:
                src = page['src']
                print(src)
                pageIndex += 1
                outFile = os.path.join(destination, f"{pageIndex:04d}")
                dlImage(src, outFile, drm)

    else:
        print('could not find pageStructure from json response')
        sys.exit(1)

    if nextReadableProductUri != None:
        print("Next URI: ", nextReadableProductUri)
else:
    print('could not find readableProduct from json response') response')