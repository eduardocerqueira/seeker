#date: 2023-03-10T16:39:52Z
#url: https://api.github.com/gists/c2150c2fdecfc3d1a35bbe64338ab2f7
#owner: https://api.github.com/users/marcosfelt

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import make_interp_spline
from typing import Callable, Dict, List, Optional, Union

def parallel_plot(
    df: pd.DataFrame,
    cols: List[str],
    color_col: str,
    log_cols: Optional[List[str]] = None,
    cmap="Spectral",
    spread=None,
    curved: bool = False,
    curvedextend: float = 0.1,
    alpha: float = 0.4,
):
    """Produce a parallel coordinates plot from pandas dataframe with line colour with respect to a column.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to plot
    cols: List[str]
        Columns to use for axes
    color_col: str
        Column to use for colorbar
    cmap: str
        Colour palette to use for ranking of lines
    spread:
        Spread to use to separate lines at categorical values
    curved: bool, optional
        Spline interpolation along lines. Default is False
    curvedextend: float, optional
        Fraction extension in y axis, adjust to contain curvature. Default is 0.1

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object
    axes: matplotlib.axes.Axes
        Axes object

    Notes
    -----
    Copied directly from: https://github.com/jraine/parallel-coordinates-plot-dataframe

    """
    colmap = cm.get_cmap(cmap)
    cols = cols + [color_col]
    log_cols = log_cols or []

    fig, axes = plt.subplots(
        1, len(cols) - 1, sharey=False, figsize=(1.5 * len(cols) + 3, 5)
    )
    valmat = np.ndarray(shape=(len(cols), len(df)))
    x = np.arange(0, len(cols), 1)
    ax_info = {}
    for i, col in enumerate(cols):
        vals = df[col]
        if ((vals.dtype == float) or (vals.dtype == int)) & (len(np.unique(vals)) > 10):
            dtype = vals.dtype
            if col in log_cols:
                vals = np.log(vals)
            minval = np.min(vals)
            maxval = np.max(vals)
            vals = np.true_divide(vals - minval, maxval - minval)
            rangeval = maxval - minval
            nticks = 5
            if rangeval < 1 and rangeval > 0.01:
                rounding = 2
            elif rangeval > 1 and rangeval < 10:
                rounding = 0
            elif rangeval > 10 and rangeval < 100:
                rounding = -1
            elif rangeval > 100 and rangeval < 1000:
                rounding = -2
            else:
                rounding = 4
            if dtype == float and col not in log_cols:
                tick_labels = [
                    round(minval + i * (rangeval / nticks), rounding)
                    for i in range(nticks + 1)
                ]
            elif dtype == int and col not in log_cols:
                tick_labels = [
                    str(int(minval + i * (rangeval // nticks)))
                    for i in range(nticks + 1)
                ]
            else:
                tick_labels = [
                    "{:.0e}".format(np.exp(minval + i * (rangeval / nticks)))
                    for i in range(nticks + 1)
                ]

            # tick_labels = clean_axis_labels(tick_labels)
            ticks = [0 + i * (1.0 / nticks) for i in range(nticks + 1)]
            valmat[i] = vals
            ax_info[col] = [tick_labels, ticks]
        else:
            vals = vals.astype("category")
            cats = vals.cat.categories
            c_vals = vals.cat.codes
            minval = -0.5
            maxval = len(cats) - 0.5
            if maxval == 0:
                c_vals = 0.5
            else:
                c_vals = np.true_divide(c_vals - minval, maxval - minval)
            tick_labels = cats
            ticks = np.unique(c_vals)
            ax_info[col] = [tick_labels, ticks]
            if spread is not None:
                offset = np.arange(-1, 1, 2.0 / (len(c_vals))) * 2e-2  # type: ignore
                np.random.shuffle(offset)
                c_vals = c_vals + offset
            valmat[i] = c_vals

    extendfrac = curvedextend if curved else 0.05
    grey = "#454545"
    for i, ax in enumerate(axes):
        remove_frame(ax, sides=["top", "bottom"])
        set_axis_color(ax, color=grey)
        ax.tick_params(colors=grey, which="both")
        for idx in range(valmat.shape[-1]):
            if curved:
                x_new = np.linspace(0, len(x), len(x) * 20)
                a_BSpline = make_interp_spline(
                    x, valmat[:, idx], k=3, bc_type="natural"
                )
                y_new = a_BSpline(x_new)
                ax.plot(x_new, y_new, color=colmap(valmat[-1, idx]), alpha=alpha)
            else:
                ax.plot(x, valmat[:, idx], color=colmap(valmat[-1, idx]), alpha=alpha)
        ax.set_ylim(0 - extendfrac, 1 + extendfrac)
        ax.set_xlim(i, i + 1)

    for dim, (ax, col) in enumerate(zip(axes, cols)):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))
        ax.set_yticklabels(ax_info[col][0])
        ax.set_xticklabels([cols[dim]])

    plt.subplots_adjust(wspace=0)
    norm = mpl.colors.Normalize(0, 1)  # *axes[-1].get_ylim())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        pad=0,
        ticks=ax_info[color_col][1],
        extend="both",
        extendrect=True,
        extendfrac=extendfrac,
    )
    # if curved:
    #     cbar.ax.set_ylim(0 - curvedextend, 1 + curvedextend)
    cbar.ax.set_yticklabels(ax_info[color_col][0])
    cbar.ax.set_xlabel(color_col, labelpad=30.0, color=grey)

    return fig, axes