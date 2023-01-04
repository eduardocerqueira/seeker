#date: 2023-01-04T16:48:44Z
#url: https://api.github.com/gists/37632b08f5a83ad913135896f08dcace
#owner: https://api.github.com/users/andy0130tw

import math
from operator import itemgetter

HEADER = '''\
# Rime dictionary
# encoding: utf-8
#
# Data converted from McBopomofo

---
name: qbane_mc_bopomofo
version: "2023.01.05"
sort: by_weight
use_preset_vocabulary: false
max_phrase_length: 6
min_phrase_weight: 100
...
'''

bpmfs = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ'
pinyins = 'b p m f d t n l g k h j q x zh ch sh r z c s i u v a o e eh ai ei ao ou an en ang eng er'.split(' ')

trans = {}
for i in range(len(bpmfs)):
    trans[bpmfs[i]] = pinyins[i]

def to_pinyin(ps):
    buf = ''
    for p in ps[:-1]:
        buf += trans[p]
    tone = 'ˊˇˋ˙'.find(ps[-1])
    if tone >= 0:
        buf += chr(0x32 + tone)
    else:
        buf += trans[ps[-1]] + '1'
    return buf

with open('mcbpmf-data.txt') as f:
    header = f.readline()
    assert header == '# format org.openvanilla.mcbopomofo.sorted\n'

    l = []

    for ln in f:
        _readings, text, _weight = ln[:-1].split(' ')
        if _readings[0] == '_':
            continue
        if _weight == '-8':
            continue

        readings = ' '.join(map(to_pinyin, _readings.split('-')))
        weight = math.floor(pow(10, float(_weight)) * 1e10)
        if weight == 0: weight = 1

        l.append((text, readings, weight))

l.sort(key=itemgetter(0))

print(HEADER)

for text, readings, value in l:
    print(f'{text}\t{readings}\t{value}')
