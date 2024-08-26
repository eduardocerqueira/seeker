#date: 2024-08-26T17:07:57Z
#url: https://api.github.com/gists/6a418f82d17c657cccc5e7967391b904
#owner: https://api.github.com/users/alonsoir

import webbrowser

with open('links.txt') as file:
    links = file.readlines()
    for link in links:
        link = link.strip()  # Remove any whitespace or newline characters
        print(link)
        webbrowser.open(link)  # Use the actual link, not the string 'link'
