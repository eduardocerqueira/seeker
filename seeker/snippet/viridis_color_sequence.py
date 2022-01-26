#date: 2022-01-26T17:14:16Z
#url: https://api.github.com/gists/821fb3694ab944aef0ad30030df9db7a
#owner: https://api.github.com/users/tommylees112

from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns

n_unique_vals = 5
colors = cm.get_cmap('viridis', n_unique_vals)(np.linspace(0, 1, n_unique_vals))
sns.palplot(colors)