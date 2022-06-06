#date: 2022-06-06T17:04:28Z
#url: https://api.github.com/gists/84d4e67e9662190ef4062a639e3eca96
#owner: https://api.github.com/users/berrybretch

import re
import pickle

# this is just so much faster than bs4


def get_span(itemprop, sauce):
    x = sauce.find(itemprop)
    y = sauce[x:].find("<")
    return sauce[x : x + y].replace(itemprop, "").replace("\\n", "").replace("\\r", "")


def get_trs(itemprop, sauce):
    x = sauce.find(itemprop)
    y = sauce[x:].find("</td")
    return sauce[x : x + y].replace(itemprop, "").replace("</th><td>", "")


blueprint = {
    "name": lambda x: get_span('"legalName">', x),
    "mailingAddress": lambda x: get_span('"streetAddress">', x),
    "mailingLocality": lambda x: get_span('"addressLocality">', x),
    "postal Code": lambda x: get_span('"postalCode">', x),
    "telephone": lambda x: get_span('"telephone">', x),
    "servicePopulation": lambda x: get_trs("Service Population", x),
    "collectionSize": lambda x: get_trs("Collection Size", x),
    "annualCirculation": lambda x: get_trs("Annual Circulation", x),
    "libraryID": lambda x: get_trs("libraries.org ID", x),
    "NCES LIBID": lambda x: get_trs("NCES LIBID", x),
}

with open("pages", "rb") as file:
    pages = pickle.load(file)


entries = []
for i in pages:
    red = {}
    for key, val in blueprint.items():
        red[key] = val(i)
    entries.append(red)

with open("entries", "wb") as file:
    pickle.dump(entries, file)