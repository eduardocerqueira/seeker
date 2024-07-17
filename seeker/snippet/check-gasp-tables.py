#date: 2024-07-17T16:47:05Z
#url: https://api.github.com/gists/a890145b50d4bfc0de1c1e482a52cd5d
#owner: https://api.github.com/users/gredler

#!/usr/bin/python3

import os
from fontTools.ttLib import TTFont

linuxHome = '/usr/share/fonts'
windowsHome = 'C:/Windows/Fonts'

for subdir, dirs, files in os.walk(windowsHome): # use windowsHome on Windows, linuxHome on Linux
    for file in files:
        if file.endswith('ttf') or file.endswith('otf'):
            path = os.path.join(subdir, file)
            font = TTFont(path)
            ranges = ''
            if 'gasp' in font.keys():
                for key in font['gasp'].gaspRange.keys(): # key is max range value in PPEM (pixels per em)
                    value = font['gasp'].gaspRange[key]
                    if value & 0x01 and value & 0x02: meaning = 'H+AA' # hinting and anti-aliasing
                    elif value & 0x01: meaning = 'H' # hinting only
                    elif value & 0x02: meaning = 'AA' # anti-aliasing only
                    else: meaning = '?'
                    ranges += '<=' + str(key) + ':' + meaning + ' '
            else:
                ranges = 'NONE '
            print(ranges + '@ ' + file)