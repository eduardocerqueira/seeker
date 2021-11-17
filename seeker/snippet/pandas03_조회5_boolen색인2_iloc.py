#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 df.iloc

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
print(df)
'''
      a     b     c     d
0     1     2     3     4
1   100   200   300   400
2  1000  2000  3000  4000
 '''

print(df.iloc[[True, False, True]])
'''
      a     b     c     d
0     1     2     3     4
2  1000  2000  3000  4000
'''
print(df.iloc[[True, False, True], [0, 1]])
'''
      a     b
0     1     2
2  1000  2000
'''

print(df.iloc[:, [True, False, True, False]])
'''
      a     c
0     1     3
1   100   300
2  1000  3000
'''

print(df.iloc[:, [True, False, True, False]])