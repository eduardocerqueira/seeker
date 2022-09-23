#date: 2022-09-23T17:15:39Z
#url: https://api.github.com/gists/436e6517258be7e6d03318a18b2e2209
#owner: https://api.github.com/users/paiv

#!/usr/bin/env python
import re


def main(infile, hmin, hmax):
    'Spec: https://kramdown.gettalong.org/converter/html.html'
    text = infile.read()
    rx = re.compile(r'^(#+)[ \t]+(.*?)\s*$', re.M)
    rl = re.compile(r'^[\W\d]+')
    ru = re.compile(r'[^\w\s-]+|_')
    rs = re.compile(r'[\s]')
    seen = dict()
    def resolve(q):
        i = seen.get(q, 0)
        seen[q] = i + 1
        return resolve(f'{q}-{i}') if i else q
    for m in rx.finditer(text):
        n,s = m[1], m[2]
        if hmin and len(n) < hmin: continue
        if hmax and len(n) > hmax: continue
        n = len(n) - (hmin or 0)
        w = '  ' * n
        t = ru.sub('', s)
        t = rl.sub('', t)
        q = rs.sub('-', t.lower()) or 'section'
        q = resolve(q)
        print(f'{w}* [{s}](#{q})')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('r'), nargs='?', default=sys.stdin, help='Markdown input')
    parser.add_argument('-n', '--min-heading-level', metavar='N', type=int, default=2, help='default 2')
    parser.add_argument('-x', '--max-heading-level', metavar='X', type=int, default=3, help='default 3')
    args = parser.parse_args()

    main(args.infile, hmin=args.min_heading_level, hmax=args.max_heading_level)
