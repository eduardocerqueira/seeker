#date: 2025-05-19T17:07:21Z
#url: https://api.github.com/gists/e05f2b650c8a8ddcb1fab576ab937c31
#owner: https://api.github.com/users/battyone

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt

size = 1e4
aa = 10**np.random.normal(-2.0, size=int(size))
aa = np.clip(aa, 1e-5, 1.0)
bb = 10**np.random.uniform(-4, 0, size=aa.size)

def logspace(vals, size):
    extr = [vals.min(), vals.max()]
    return np.logspace(*np.log10(extr), size)

xedges = logspace(aa, 20)
yedges = logspace(bb, 25)

hist, *_ = sp.stats.binned_statistic_2d(aa, bb, None, bins=(xedges, yedges), statistic='count')
norm = mpl.colors.LogNorm(vmin=hist[hist>0].min(), vmax=hist.max())

ax = plt.gca()
ax.set(xscale='log', yscale='log')

xx, yy = np.meshgrid(xedges, yedges)

pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap='viridis', norm=norm)
ax.scatter(aa, bb, s=2, alpha=0.1, color='r')

plt.colorbar(pcm, ax=ax, label='number')