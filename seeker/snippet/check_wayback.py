#date: 2022-04-22T17:09:58Z
#url: https://api.github.com/gists/4ef1a23feb60f456d7a4fbaf4da28e7f
#owner: https://api.github.com/users/edsu

#!/usr/bin/env python3

# This is an example of seeing what unique HTML webpages there are in the
# Wayback Machine for the http://myshtetl.org/ website after 2022-03-01.

from wayback import WaybackClient

wb = WaybackClient()

pages = set()
for rec in wb.search('http://myshtetl.org/', matchType='prefix', from_date='2022-03-01'):
    if 'html' in rec.mime_type and rec.status_code == 200 and rec.url not in pages:
        pages.add(rec.url)
        print(rec.url)



