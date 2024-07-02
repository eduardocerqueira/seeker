#date: 2024-07-02T16:33:23Z
#url: https://api.github.com/gists/f18bb36f6feab744da04499d2d04b62e
#owner: https://api.github.com/users/joeyv120

# https://martechwithme.com/get-list-pages-url-from-website-python/
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url#39225272
# https://stackoverflow.com/questions/38489386/how-to-fix-403-forbidden-errors-when-calling-apis-using-python-requests


import requests
import os
import re
from usp.tree import sitemap_tree_for_homepage
from time import sleep


def list_pages(url):
    listPagesRaw = []
    tree = sitemap_tree_for_homepage(url)
    for page in tree.all_pages():
        listPagesRaw.append(page.url)
    # Go through List Pages Raw output a list of unique pages links
    listPages = []
    for page in listPagesRaw:
        if page in listPages:
            pass
        else:
            listPages.append(page)
    return listPages


def findAndDownload(page, folder):
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0'}
    response = requests.get(page, headers=header)
    txt = response.text
    pattern = r"https://yumdm.com/(.*)/(.*).pdf"
    try:
        result = re.finditer(pattern, txt)
    except AttributeError:
        print("Error: " + page)
        return None
    for match in result:
        fURL = match.group(0)
        fName = match.group(2) + '.pdf'
        print(fURL)
        print(fName)
        response = requests.get(fURL, headers=header, stream=True)
        if os.path.isfile(folder + fName):
            print('File already exists.')
            return None
        else:
            print('Downloading...')
            with open(folder + fName, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)


if __name__ == '__main__':
    bURL = "https://yumdm.com/"
    folder = os.path.expanduser('~') + '\\Downloads\\YUMDM\\'
    if not os.path.exists(folder):
        os.mkdir(folder)

    pages = list_pages(bURL)
    for page in pages[0:5]:  # just grab the 5 most recent posts
        print(str(pages.index(page) + 1) + " of " + str(len(pages)))
        findAndDownload(page, folder)
        sleep(1)  # Don't bug them too much
    print('complete')
