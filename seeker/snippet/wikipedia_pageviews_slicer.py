#date: 2021-09-27T16:47:44Z
#url: https://api.github.com/gists/9b37f420278f8bc60b9a6bcc7bd5b6b7
#owner: https://api.github.com/users/tmanabe

from argparse import ArgumentParser
from bz2 import open

"""
Usage: python wikipedia_pageviews_slicer.py --prefix ja.wikipedia pageviews-202108-user.bz2 pageviews-202108-user-ja.wikipedia.bz2
"""

parser = ArgumentParser()
parser.add_argument('-e', '--encoding', default='utf-8')
parser.add_argument('-p', '--prefix', default='simple.wikipedia')
parser.add_argument('source')
parser.add_argument('target')
args = parser.parse_args()

with open(args.source, 'rt', encoding=args.encoding) as bz2fs:
  with open(args.target, 'wt', encoding=args.encoding) as bz2ft:
    for l in bz2fs:
      if l.startswith(args.prefix):
        bz2ft.write(l)
