#date: 2022-11-04T17:16:07Z
#url: https://api.github.com/gists/a511daa9d82c40c00d8baf96db507930
#owner: https://api.github.com/users/guidoism

import json, subprocess, serial, re, rich, rich.console, os
updated = os.stat('layers.json').st_mtime
layers_in = json.load(open('layers.json'))

col_color = {
    0: '[cyan]',
    1: '[bold cyan]',
    2: '[bold magenta1]',
    3: '[bold green1]',
    4: '[bold turquoise2]',
    5: '[turquoise2]',
    6: '[turquoise2]',
    7: '[bold turquoise2]',
    8: '[bold green1]',
    9: '[bold magenta1]',
    10: '[bold cyan]',
    11: '[cyan]',
}

layers = []
for l in layers_in:
    rows = []
    for r in l.split('\n'):
        row = []
        for i, m in enumerate(re.findall(r'(\s*\S+\s*)', r)):
            if '◌' in m:
                row.append('[dim]')
                row.append(m)
                row.append('[/]')
            else:
                row.append(col_color[i])
                row.append(m)
                row.append('[/]')
        rows.append(''.join(row))
    layers.append('\n'.join(rows))
        
p = subprocess.run(['discotool', 'json'], capture_output=True)
devs = json.loads(p.stdout)
vol = lambda d: {v['name'] for v in d['volumes']}
path = [d['ports'][0]['dev'] for d in devs if 'KEEB-1' in vol(d)][0]
ser = serial.Serial(path)

con = rich.console.Console()
while s := ser.readline():
    if m := re.match(r'Layer: (\d+)', s.decode()):
        n = int(m.group(1))
        con.clear()
        con.print(layers[n])

        if os.stat('layers.json').st_mtime > updated:
            updated = os.stat('layers.json').st_mtime
            layers = json.load(open('layers.json'))
