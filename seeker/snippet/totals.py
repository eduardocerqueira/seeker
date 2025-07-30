#date: 2025-07-30T17:09:50Z
#url: https://api.github.com/gists/638162bec37de32379d71f5365635439
#owner: https://api.github.com/users/qwolphin

import sys
from decimal import Decimal, getcontext, ROUND_UP

getcontext().rounding = ROUND_UP

text = ''.join(sys.stdin)
persons = text.split('\n\n')

for record in persons:
    record = record.removesuffix('\n')
    header, *items = record.split('\n')

    total = Decimal(0)
    for item in items:
        segments = item.split(' ')
        if segments[-1].startswith('x'):
            count = int(segments[-1][1:])
            price_raw = segments[-2]
        else:
            count = 1
            price_raw = segments[-1]

        if price_raw.endswith('k'):
            price = Decimal(price_raw[:-1]) * 1000
        else:
            price = Decimal(price_raw)

        total += price * count

    total_USD = total / 1335
    total_UZS = int(round(total_USD * 12600, -4)/1000)
    total_USD = round(total_USD, 0)

    print(record)
    print(f'Всего: ${total_USD} или {total_UZS:,}k сум\n')