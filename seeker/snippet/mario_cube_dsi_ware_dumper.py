#date: 2024-09-20T16:38:55Z
#url: https://api.github.com/gists/d362c8a0402344726cc92d3d4133d8e8
#owner: https://api.github.com/users/p4p1

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Made by papi
# Created on: Fri 20 Sep 2024 04:36:32 PM IST
# mario_cube_dsi_ware_dumper.py
# Description:
#  This python script will download all of the nds file present on the
#  dsi ware section of the mario cube website. All credit goes to them this
#  is just a little script to make downloading files there easier.

import requests
from bs4 import BeautifulSoup

URL="https://repo.mariocube.com/DSiWare/NDS/"

def dl_file(url, file_name):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            for chunck in response.iter_content(chunk_size=8192):
                file.write(chunck)
        return True
    return False

let = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' ]

for letter in let:
    response = requests.get(URL + letter + "/")

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        href_links = [link['href'] for link in links]
        i = 0

        for l in href_links:
            if i % 2 == 0:
                i += 1
                continue
            i += 1
            if "USA" in l:
                if dl_file(URL + letter + "/" + l,l.replace('%20', ' ')):
                    print("done: %s" % l.replace('%20', ' '))
