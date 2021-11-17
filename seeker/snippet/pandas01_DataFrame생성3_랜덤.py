#date: 2021-11-17T16:50:45Z
#url: https://api.github.com/gists/37938448a25841c327b50e28790b1231
#owner: https://api.github.com/users/chinshin5513

import numpy as np
import pandas as pd  # 주로 두 개 패키지를 깔고 감

arr = np.random.random((3,3))
print(arr)  # arr 중첩 리스트로 생성
'''
[[0.08657611 0.11206673 0.27232677]
 [0.13737225 0.83165976 0.02187929]
 [0.45609119 0.63178951 0.17021176]]
'''

df = pd.DataFrame(arr,
                  index=[1,2,3],
                  columns=["랜덤값1", "랜덤값2", "랜덤값3"])
print(df)
'''
       랜덤값1      랜덤값2      랜덤값3
1  0.086576  0.112067  0.272327
2  0.137372  0.831660  0.021879
3  0.456091  0.631790  0.170212

 '''