#date: 2021-12-07T17:20:21Z
#url: https://api.github.com/gists/f69190613c850c84f19f84125fa5cf24
#owner: https://api.github.com/users/kaathewise

import re
import sys
from collections import Counter
from itertools import chain

print((lambda l: sum(1 for v in Counter(chain.from_iterable((lambda k: (((x1*i+x2*(k-i))//k, (y1*i+y2*(k-i))//k) for i in range(k+1)))(max(abs(x1-x2), abs(y1-y2))) for x1, y1, x2, y2 in l)).values() if v>1))(tuple(map(int, re.findall(r'\d+', l))) for l in sys.stdin.readlines()))