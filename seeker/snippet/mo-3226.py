#date: 2025-06-20T17:12:56Z
#url: https://api.github.com/gists/f2a9299d913dc9801c639134db6e06e8
#owner: https://api.github.com/users/manzt

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(np, plt):
    def make_plot(with_figure: bool):
        if with_figure:
            plt.figure()
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([0, 2, 1, 4, 3, 5])

        plt.plot(x, y)
        plt.xlabel("X-axis Label")
        plt.ylabel("Y-axis Label")
        plt.title("Simple Line Plot Example")

        plt.show()

    make_plot(with_figure=True)
    return (make_plot,)


@app.cell
def _(make_plot):
    make_plot(with_figure=False)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


if __name__ == "__main__":
    app.run()
