#date: 2025-01-03T17:07:45Z
#url: https://api.github.com/gists/cff3c34852021e42c0c91bae38b10d37
#owner: https://api.github.com/users/PieroPaialungaAI

import numpy as np
import matplotlib.pyplot as plt
X_1 = np.linspace(0,1,100).reshape(-1,1)
noise = np.random.normal(0,0.08,X_1.shape)
X_2 = X_1+noise
plt.scatter(X_1,X_2,s=20)
plt.xlabel(r'$X_1$',fontsize=12)
plt.ylabel(r'$X_2$',fontsize=12)