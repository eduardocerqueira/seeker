#date: 2024-01-17T17:05:32Z
#url: https://api.github.com/gists/a01ca686919a0857c2830601cabfdea2
#owner: https://api.github.com/users/MalcolmMielle

# This script will create a graph with a light grey grid to see the spacing between values
# The axis will be visible only on the bottom and left, and both the axis and labels will
# be light blue/green

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": [],
        "font.size": 25,
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharey=True)
def axis_style(ax):
    ax.spines["bottom"].set_color("#0d6a82")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#0d6a82")
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(axis="both", colors="#0d6a82", labelsize=80)
    ax.grid(color="grey", linestyle="--", linewidth=2, alpha=0.3)
    ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
    ax.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_ylim(bottom=-2)
    ax.set_xlim(left=0, right=4)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Error (m)")
    ax.xaxis.label.set_color("#0d6a82")
    ax.yaxis.label.set_color("#0d6a82")


axis_style(ax[0])
axis_style(ax[1])