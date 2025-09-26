#date: 2025-09-26T16:55:49Z
#url: https://api.github.com/gists/56b61640b25fef79c64f9736605cc7ce
#owner: https://api.github.com/users/AspirantDrago

from math import *

FILENAME = 'function.txt'
function_str = input()
with open(FILENAME, 'w') as f:
    for x_int in range(200 + 1):
        x = x_int / 100
        y = round(eval(function_str), 3)
        print(x, y, sep='\t', file=f)
