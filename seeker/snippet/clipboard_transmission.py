#date: 2024-09-11T17:05:15Z
#url: https://api.github.com/gists/c7bcf8072a2322927b48ed16e9a74d92
#owner: https://api.github.com/users/robmcelhinney

from transmission_rpc import Client
import time
import pyperclip
import re
import os

tc = Client(host=os.getenv('T_HOST'), 
            port=os.getenv('T_PORT'), 
            username=os.getenv('T_USERNAME'), 
            password= "**********"

magnet_link = ''

def is_magnet_link(string):
    # Magnet URIs follow this pattern: "magnet:?xt=urn:btih:<hash>"
    return bool(re.match(r'^magnet:.*xt=urn:btih:[a-fA-F0-9]{35}.*', string))

def is_new_link(link, paste):
    return link != paste

def add_torrent():
    global magnet_link
    paste = pyperclip.paste()

    is_new = is_new_link(magnet_link, paste)
    magnet_link = paste
    if not is_new:
        print("Skipping non-new link.")
        return

    if not is_magnet_link(magnet_link):
        print("Skipping non link.")
        return

    # Query existing torrents to check for duplicates (if necessary)
    active_torrents = [t for t in tc.get_torrents() if t.status == 'downloading' or t.status == 'pending']

    # Check for duplicate magnet links
    if magnet_link not in active_torrents:
        print(f"Adding torrent: {magnet_link}")
        tc.add_torrent(magnet_link)
    else:
        print(f"Duplicate magnet link detected: {magnet_link}")

def main():
    delay = 5

    while True:
        add_torrent()
        time.sleep(delay)

if __name__ == '__main__':
    main()
    main()
