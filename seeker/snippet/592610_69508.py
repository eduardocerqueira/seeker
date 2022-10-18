#date: 2022-10-18T17:13:07Z
#url: https://api.github.com/gists/25b7639fd2b8fd1078c9e2d37e9c7a7f
#owner: https://api.github.com/users/galenseilis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

y = np.abs(np.concatenate((np.random.normal(size=1000), np.random.normal(loc=10**10,size=1000))))
x = ['A']*1000 + ['B']*1000

d = {'Condition':x, 'Response':y}

df = pd.DataFrame(d)

sns.violinplot(x='Condition', y='Response', data=df)
plt.yscale('log')
plt.show()