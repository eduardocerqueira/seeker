#date: 2021-09-01T17:00:45Z
#url: https://api.github.com/gists/4c2686e5f3a850ceb972546c37f204a9
#owner: https://api.github.com/users/innocentwt

import requests
from bs4 import BeautifulSoup as bs4

root = "https://pfps.gg"
last_query = ""

def parse_input(value):
    return [value.split(" ")[0], " ".join(value.split(" ")[1:])]

def get_page(url):
    r = requests.get(url)
    return r.content

def scrape_page(url):
    pfps = []
    soup = bs4(get_page(url), "html.parser")
    scraped_pfps = soup.find_all("div", {"class": ["item-details", "text-center"]})
    for pfp in scraped_pfps:
        child = pfp.findChildren("a", recursive=False)
        if child:
            pfps.append(child[0])
    return pfp_dict(pfps)

def pfp_dict(l):
    pfps = {}
    for pfp in l:
        pfp = str(pfp)
        if pfp.split('href="')[1].split('">')[0].split("/")[-1][0:4].isdigit():
            pfps[pfp.split('>')[1].split("</")[0]] = pfp.split('href="')[1].split('">')[0]
    return pfps

def cmd_list(path):
    global last_query
    last_query = path
    dic = scrape_page(root + f"/pfps/{path}")
    for k, v in dic.items():
        print(f"[{k}] {v[20:]}")

def cmd_download(url=None):
    if url:
        doc = str(get_page(f"{root}/pfp/{url}"))
        file_url = doc.split('" class="btn btn-block btn-success" download>')[0].split('<a href="')[-1]
        filename = f"{url}{file_url[-4:]}"
        r = requests.get(file_url)
        open(filename, "wb").write(r.content)
        print(f"Saved to {filename}")
    else:
        dic = scrape_page(f"{root}/pfps/{last_query}")
        for k, v in dic.items():
            cmd_download(url=v.split("/")[-1])

def cmd_quit(arg):
    quit()

def console():
    while True:
        cmd = parse_input(input("> "))
        eval(f'cmd_{cmd[0]}("{cmd[1]}")')

console()
