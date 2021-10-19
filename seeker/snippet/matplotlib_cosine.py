#date: 2021-10-19T16:54:05Z
#url: https://api.github.com/gists/44482e2bddecb78013c219e80423a815
#owner: https://api.github.com/users/hankyojeong

import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

x = np.linspace(-np.pi, np.pi, 128)
y = np.cos(x)
plt.plot(y)