#date: 2023-08-03T17:01:03Z
#url: https://api.github.com/gists/f432b5b0af930f0d894d3af8204d0e12
#owner: https://api.github.com/users/DariaVesela

# you need beautifulsoup4 and google installed
from googlesearch import search


def search_google(query, tld="com", num=50, stop=50, pause=2):


    search_results = []
    for result in search(query, tld=tld, num=num, stop=stop, pause=pause):
        search_results.append(result)

    return search_results