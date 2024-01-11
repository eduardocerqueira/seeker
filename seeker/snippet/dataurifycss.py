#date: 2024-01-11T17:03:43Z
#url: https://api.github.com/gists/76af86f6095a53a67080d31579610176
#owner: https://api.github.com/users/vadimkantorov

# python3 dataurifycss.py -i assets/main.3cfd67ba.css --delete > assets/main.css

import os
import argparse
import re
import sys
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--input-css-path', '-i')
parser.add_argument('--root-dir')
parser.add_argument('--delete', action = 'store_true')
parser.add_argument('--mime', action = 'append', default = ['.svg=image/svg+xml', '.jpg=image/jpeg', '.png=image/png', '.gif=image/gif', '.woff2=font/woff2', '.woff=font/woff', '.ttf=font/ttf'])
args = parser.parse_args()

mime = dict(m.split('=') for m in args.mime)
css = open(args.input_css_path).read()
root_dir = args.root_dir or '.'

seen = set()

def dataurify(s):
    o = s.group(0)
    g = s.group(1).strip("'" + '"')
    e = os.path.splitext(g)[-1]
    p = os.path.join(root_dir, g.lstrip('/')) if g.startswith('/') else g
    if g.startswith('data:'):
        return g
    elif e in mime:
        t = 'url(data:{mime};base64,{encoded})'.format(mime = mime[e], encoded = base64.b64encode(open(p, 'rb').read()).decode())
        seen.add(p)
        return t
    else:
        return o


res = re.sub('url\((.+?)\)', dataurify, css)

print('\n'.join(sorted(seen)), file = sys.stderr)
for p in (seen if args.delete else []):
    os.remove(p)

print(res)