#date: 2022-07-21T17:11:05Z
#url: https://api.github.com/gists/a77af7e4554de3042440fda8cfc10ce9
#owner: https://api.github.com/users/jugmac00

"""
Usage:

- create and activate a virtual env
- `pip install launchpadlib`
- run `python main.py https://launchpad.net/~deadsnakes`

API:
https://launchpad.net/+apidoc/devel.html

Getting started:
https://help.launchpad.net/API/launchpadlib
"""

from argparse import ArgumentParser
from urllib.parse import urlparse

from launchpadlib.launchpad import Launchpad

parser = ArgumentParser()
parser.add_argument("team", help="Full URL to a team")
args = parser.parse_args()

cachedir = "/home/jugmac00/.launchpadlib/cache/"
launchpad = Launchpad.login_anonymously(
    "testing", "production", cachedir, version="devel"
)

if args.team:
    team = launchpad.load(urlparse(args.team).path)

    ppas = team.ppas

    for ppa in ppas:
        for ps in ppa.getPublishedBinaries():
            print(ps.display_name)
