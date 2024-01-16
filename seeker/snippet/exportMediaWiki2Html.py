#date: 2024-01-16T16:50:36Z
#url: https://api.github.com/gists/17b43ef3fc7bc6af51e4a4242111f667
#owner: https://api.github.com/users/saurabh2590

#!/usr/bin/python3

# Author: Timotheus Pokorra <timotheus.pokorra@solidcharity.com>
# source hosted at https://github.com/SolidCharity/exportMediaWiki2HTML
# licensed under the MIT license
# Copyright 2020-2021 Timotheus Pokorra

import os
from urllib import parse
import requests
import json
import re
from pathlib import Path
import argparse

description = """
Export MediaWiki pages to HTML
Call like this:
   ./exportMediaWiki2Html.py --url=https://mywiki.example.org

   Optionally pass the page id of the page you want to download, eg. for debugging:
   ./exportMediaWiki2Html.py --url=https://mywiki.example.org --page=180

   Optionally pass the page id of the category, all pages with that category will be exported:
   ./exportMediaWiki2Html.py --url=https://mywiki.example.org --category=22

   Optionally pass the namespace id, only pages in that namespace will be exported:
   ./exportMediaWiki2Html.py --url=https://mywiki.example.org --namespace=0

 "**********"  "**********"  "**********"  "**********"O "**********"p "**********"t "**********"i "**********"o "**********"n "**********"a "**********"l "**********"l "**********"y "**********"  "**********"p "**********"a "**********"s "**********"s "**********"  "**********"t "**********"h "**********"e "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
   ./exportMediaWiki2Html.py --url=https: "**********"

   Optionally pass the directory to dump the export to:
   ./exportMediaWiki2Html.py --url=https://mywiki.example.org --outputDir=export
"""
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-l','--url', help='The url of the wiki',required=True)
parser.add_argument('-u','--username', help='Your username and bot name, eg. "myuser@botname"',required=False)
parser.add_argument('-p','--password', help= "**********"=False)
parser.add_argument('-c','--category', help='The category to export',required=False)
parser.add_argument('-g','--page', help='The page to export',required=False)
parser.add_argument('-s', '--namespace', help='The namespace to export', required=False)
parser.add_argument('-n', '--numberOfPages', help='The number of pages to export, or max', required=False, default=500)
parser.add_argument('-o', '--outputDir', help='The destination directory for the export', type=Path, required=False, default="export")
parser.add_argument('--shortUrl', help='Custom short url path for the wiki', required=False, default='wiki/')
parser.add_argument('--listPages', help='List available pages', required=False, default=False, action='store_true')
parser.add_argument('--dontOverwrite', help='Skip already downloaded files', required=False, default=False, action='store_true')
try:
    parser.add_argument('--ssl', help='Enable SSL redirection', required=False, default=True, action=argparse.BooleanOptionalAction)
except AttributeError:
    # BooleanOptionalAction was introduced in Python 3.9
    parser.add_argument('--ssl', help='Enable SSL redirection', required=False, default=True)
args = parser.parse_args()

if args.numberOfPages != "max":
  try:
    int(args.numberOfPages)
    numberOfPages = str(args.numberOfPages)
  except ValueError:
      print("Provided number of pages is invalid")
      exit(-1)
else:
  numberOfPages = "max"

url = args.url
if not url.endswith('/'):
  url = url + '/'
# get the subpath of the url, eg. https://www.example.org/wiki/ => wiki/, or empty for no subpath
subpath = url[url.index("://") + 3:]
subpath = subpath[subpath.index("/")+1:]

pageOnly = -1
categoryOnly = -1
namespace = args.namespace
if args.category is not None:
  categoryOnly = int(args.category)
  if namespace is None:
    namespace = "*" # all namespaces
else:
  if namespace is None:
    namespace = 0
  # the allpages API only supports integer IDs
  namespace = str(int(namespace))
if args.page is not None:
  pageOnly = int(args.page)

(args.outputDir / "img").mkdir(parents=True, exist_ok=True)

if not args.shortUrl.endswith('/'):
  args.shortUrl = args.shortUrl + '/'
shortUrl = args.shortUrl 

S = requests.Session()

 "**********"i "**********"f "**********"  "**********"a "**********"r "**********"g "**********"s "**********". "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"  "**********"i "**********"s "**********"  "**********"n "**********"o "**********"t "**********"  "**********"N "**********"o "**********"n "**********"e "**********"  "**********"a "**********"n "**********"d "**********"  "**********"a "**********"r "**********"g "**********"s "**********". "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"s "**********"  "**********"n "**********"o "**********"t "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
  LgUser = args.username
  LgPassword = "**********"

  # Retrieve login token first
  PARAMS_0 = {
      'action':"query",
      'meta': "**********"
      'type':"login",
      'format':"json"
  }
  R = S.get(url=url + "/api.php", params=PARAMS_0)
  DATA = R.json()
  LOGIN_TOKEN = "**********"

  # Main-account login via "action=login" is deprecated and may stop working without warning. To continue login with "action=login", see [[Special: "**********"
  PARAMS_1 = {
      'action':"login",
      'lgname':LgUser,
      'lgpassword': "**********"
      'lgtoken': "**********"
      'format':"json"
  }

  R = S.post(url + "/api.php", data=PARAMS_1)
  try:
    DATA = R.json()
  except:
    print("cannot parse json from action:login")
    print(R.content)
    exit(-1)
  if "error" in DATA:
    print(DATA)
    exit(-1)

if categoryOnly != -1:
  params_all_pages = {
    'action': 'query',
    'list': 'categorymembers',
    'format': 'json',
    'cmpageid': categoryOnly,
    'cmnamespace': namespace,
    'cmlimit': numberOfPages
  }
else:
  params_all_pages = {
    'action': 'query',
    'list': 'allpages',
    'format': 'json',
    'apnamespace': namespace,
    'aplimit': numberOfPages
  }

response = S.get(url + "api.php", params=params_all_pages)
data = response.json()

if "error" in data:
  print(data)
  if data['error']['code'] == "readapidenied":
    print()
    print("get login token here: "**********"
    print("and then call this script with parameters: "**********"
    exit(-1)
if categoryOnly != -1:
  pages = data['query']['categorymembers']
else:
  pages = data['query']['allpages']

# user may want to download a single page, but needs to know the page number
if args.listPages:
    for page in pages:
        print(f'{page["pageid"]}: {page["title"]}')
    exit(0)

while 'continue' in data and (numberOfPages == 'max' or len(pages) < int(numberOfPages)):
  if categoryOnly != -1:
    params_all_pages['cmcontinue'] = data['continue']['cmcontinue']
  else:
    params_all_pages['apcontinue'] = data['continue']['apcontinue']

  response = S.get(url + "api.php", params=params_all_pages)

  data = response.json()

  if "error" in data:
    print(data)
    if data['error']['code'] == "readapidenied":
      print()
      print(f'get login token here: "**********"
      print("and then call this script with parameters: "**********"
      exit(-1)

  if categoryOnly != -1:
    pages.extend(data['query']['categorymembers'])
  else:
    pages.extend(data['query']['allpages'])

def quote_title(title):
  return parse.quote(page['title'].replace(' ', '_'))

downloadedimages = []
def DownloadImage(filename, urlimg, ignorethumb=True):
  fileOut = f'{args.outputDir}/img/{filename}'
  if not filename in downloadedimages:
    if ignorethumb and '/thumb/' in urlimg:
      urlimg = urlimg.replace('/thumb/', '/')
      urlimg = urlimg[:urlimg.rindex('/')]
    if not urlimg.startswith("http"):
        urlimg = url + urlimg[1:]
    print(f"Downloading {urlimg}")
    response = S.get(urlimg)
    if response.status_code == 404:
      pass 
      #raise Exception("404: cannot download " + urlimg)
    content = response.content
    f = open(fileOut, "wb")
    f.write(content)
    f.close()
    downloadedimages.append(filename)

def DownloadFile(filename, urlfilepage):
  fileOut = f'{args.outputDir}/img/{filename}'
  if args.dontOverwrite and os.path.exists(fileOut):
      print(f'Ignoring {filename} (already downloaded)')
      downloadedimages.append(filename)
      return
  if not filename in downloadedimages:
    # get the file page
    response = S.get(urlfilepage)
    content = response.text
    filepos = content.find('href="/' + subpath + 'images/')
    if filepos == -1:
      return
    fileendquote = content.find('"', filepos + len('href="'))
    urlfile = content[filepos+len('href="') + len(subpath):fileendquote]
    DownloadImage(filename, urlfile)

def PageTitleToFilename(title):
    temp = re.sub('[^A-Za-z0-9\u0400-\u0500\u4E00-\u9FFF]+', '_', title)
    #return temp.replace("(","_").replace(")","_").replace("__", "_")
    return title

for page in pages:
    if (pageOnly > -1) and (page['pageid'] != pageOnly):
        continue
    print(page)
    quoted_pagename = quote_title(page['title'])
    url_page = url + "index.php?title=" + quoted_pagename + "&action=render"
    response = S.get(url_page)
    content = response.text
    url_title = url + "index.php?title="
    if (url_title not in content) and args.ssl:
        url_title = url_title.replace("http://", "https://")

    # in case we have links like a href="//wiki.example.org/index.php..."
    if url_title not in content:
        protocol = url_title[:url_title.index(":")]
        url_title_without_protocol = url_title[url_title.index('/'):]
        content = content.replace(f'a href="{url_title_without_protocol}', f'a href="{protocol}:{url_title_without_protocol}')

    # in case we have links like a href="//wiki.example.org/wiki/..."
    if url_title not in content:
        url_title_without_indexphp = url_title.replace("index.php?title=", shortUrl)
        content = content.replace(f'a href="{url_title_without_indexphp}', f'a href="{url_title}')

    pos = 0
    while url_title in content:
        pos = content.find(url_title)
        posendquote = content.find('"', pos)
        file_url = content[pos:posendquote]
        linkedpage = file_url
        linkedpage = linkedpage[linkedpage.find('=') + 1:]
        linkedpage = linkedpage.replace('%27', '_')
        if linkedpage.startswith('File:') or linkedpage.startswith('Datei:') or linkedpage.startswith('Image:'):
          if linkedpage.startswith('File:'):
              linkType = "File"
          elif linkedpage.startswith('Datei:'):
              linkType = "Datei"
          elif linkedpage.startswith('Image:'):
              linkType = "Image"
          origlinkedpage = linkedpage[linkedpage.find(':')+1:]
          linkedpage = parse.unquote(origlinkedpage)

          if linkType == "File" or linkType == "Datei":
            DownloadFile(linkedpage, file_url)

          # images are only downloaded for "img src="
          # we just replace the link here
          content = content.replace(url_title+linkType+":"+origlinkedpage, "img/"+origlinkedpage)

        elif "&amp;action=edit&amp;redlink=1" in linkedpage:
          content = content[:pos] + "page_not_existing.html\" style='color:red'" + content[posendquote+1:]
        elif "#" in linkedpage:
          linkWithoutAnchor = linkedpage[0:linkedpage.find('#')]
          linkWithoutAnchor = PageTitleToFilename(linkWithoutAnchor)
          content = content[:pos] + linkWithoutAnchor + ".html#" + linkedpage[linkedpage.find('#')+1:] + content[posendquote:]
        else:
          linkedpage = PageTitleToFilename(parse.unquote(linkedpage))
          content = content[:pos] + linkedpage + ".html" + content[posendquote:]

    # replace all <a href="<url>/<subpath>/images"
    imgpos = 0
    while imgpos > -1:
        imgpos = content.find('href="' + url + 'images/', imgpos)
        if imgpos > -1:
          imgendquote = content.find('"', imgpos + len('href="'))
          imgpath = content[imgpos+len('href="'):imgendquote]
          filename = imgpath[imgpath.rindex("/")+1:]
          DownloadImage(filename, imgpath, ignorethumb=False)
          content = content.replace(content[imgpos + len('href="'):imgendquote], "img/"+filename)


    # replace all <img src="/<subpath>/images"
    imgpos = 0
    while imgpos > -1:
        imgpos = content.find('src="/' + subpath + 'images/', imgpos)
        if imgpos > -1:
          imgendquote = content.find('"', imgpos + len('src="'))
          imgpath = content[imgpos+len('src="') + len(subpath):imgendquote]
          filename = imgpath[imgpath.rindex("/")+1:]
          DownloadImage(filename, imgpath, ignorethumb=False)
          content = content.replace("/"+subpath+imgpath[1:], "img/"+filename)

    # replace all srcset="/<subpath>/images..., /<subpath>/images...""
    imgpos = 0
    while imgpos > -1:
        imgpos = content.find('srcset="/' + subpath + 'images/', imgpos)
        if imgpos > -1:
          imgendquote = content.find('"', imgpos + len('srcset="'))
          srcsetval = content[imgpos+len('srcset="'):imgendquote]
          for srcsetitem in srcsetval.split(','):
            imgpath = srcsetitem.strip().split()[0][len(subpath):]
            filename = imgpath[imgpath.rindex("/")+1:]
            DownloadImage(filename, imgpath, ignorethumb=False)
            content = content.replace("/"+subpath+imgpath[1:], "img/"+filename)

    #content = content.replace('<div class="mw-parser-output">'.encode("utf8"), ''.encode("utf8"))
    content = re.sub("(<!--).*?(-->)", '', content, flags=re.DOTALL)

    f = open(args.outputDir / (PageTitleToFilename(page['title']) + ".html"), "wb")
    f.write(("<html>\n<head><meta encoding='UTF-8'/><title>" + page['title'] + "</title></head>\n<body>\n").encode("utf8"))
    f.write(("<h1>" + page['title'] + "</h1>").encode("utf8"))
    f.write(content.encode('utf8'))
    f.write("</body></html>".encode("utf8"))
    f.close()

f = open(args.outputDir / "page_not_existing.html", "wb")
f.write(("<html>\n<head><title>This page does not exist yet</title></head>\n<body>\n").encode("utf8"))
f.write(("<h1>This page does not exist yet</h1>").encode("utf8"))
f.write("</body></html>".encode("utf8"))
f.close()