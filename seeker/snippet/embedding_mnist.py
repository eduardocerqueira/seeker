#date: 2025-07-16T17:16:49Z
#url: https://api.github.com/gists/4d9e4f229f77a2222a9c20dc41eb284e
#owner: https://api.github.com/users/akshayka

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.2",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "pandas==2.3.1",
#     "polars==1.31.0",
#     "pyarrow==20.0.0",
#     "pymde==0.2.3",
#     "sqlglot==27.0.0",
#     "torch==2.7.1",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo

    import altair as alt
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymde
    import torch
    import polars as pl

    mnist = pymde.datasets.MNIST()


@app.function
def compute_embedding(embedding_dim, constraint=None, quadratic=False):
    mo.output.append(
        mo.md("Your embedding is being computed ... hang tight!").callout(
            kind="warn"
        )
    )

    constraint = constraint if constraint is not None else pymde.Standardized()

    mde = pymde.preserve_neighbors(
        mnist.data,
        attractive_penalty=pymde.penalties.Log1p
        if not quadratic
        else pymde.penalties.Quadratic,
        repulsive_penalty=pymde.penalties.Log if not quadratic else None,
        embedding_dim=embedding_dim,
        constraint=constraint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )
    X = mde.embed(verbose=True)
    mo.output.clear()
    return X, mde.distortions()


@app.function
def dataframe_from_embedding(embedding, samples=20000):
    indices = torch.randperm(mnist.data.shape[0])[:samples]
    embedding_np = embedding.numpy()[indices]

    df = pd.DataFrame(
        {
            "index": indices,
            "x": embedding_np[:, 0],
            "y": embedding_np[:, 1],
            "digit": mnist.attributes["digits"].numpy()[indices],
        }
    )
    return df


@app.function
def scatter(df, size=4):
    return (
        alt.Chart(df)
        .mark_circle(size=size)
        .encode(
            x=alt.X("x:Q").scale(domain=(-2.5, 2.5)),
            y=alt.Y("y:Q").scale(domain=(-2.5, 2.5)),
            color=alt.Color("digit:N"),
        )
        .properties(width=500, height=500)
    )


@app.function
def show_images(indices, max_images=10):
    indices = indices[:max_images]
    images = mnist.data.reshape((-1, 28, 28))[indices]
    fig, axes = plt.subplots(1, len(indices))
    fig.set_size_inches(12.5, 1.5)
    if len(indices) > 1:
        for im, ax in zip(images, axes.flat):
            ax.imshow(im, cmap="gray")
            ax.set_yticks([])
            ax.set_xticks([])
    else:
        axes.imshow(images[0], cmap="gray")
        axes.set_yticks([])
        axes.set_xticks([])
    plt.tight_layout()
    return fig


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""# Embedding MNIST""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    Here's an **embedding of MNIST**: each point represents a digit,
    with similar digits close to each other.
    """
    )
    return


@app.cell(hide_code=True)
def _(args):
    quadratic = mo.ui.switch(value=args.quadratic, label="Spectral embedding?")
    quadratic
    return (quadratic,)


@app.cell(hide_code=True)
def _(df):
    chart = mo.ui.altair_chart(scatter(df))
    chart
    return (chart,)


@app.cell(hide_code=True)
def _(chart):
    table = mo.ui.table(chart.value)
    return (table,)


@app.cell(hide_code=True)
def _(chart, table):
    # mo.stop() prevents this cell from running if the chart has
    # no selection
    mo.stop(not len(chart.value))

    # show 10 images: either the first 10 from the selection, or the first ten
    # selected in the table
    selected_images = (
        show_images(list(chart.value["index"]))
        if not len(table.value)
        else show_images(list(table.value["index"]))
    )

    mo.md(
        f"""
        **Here's a preview of the images you've selected**:

        {mo.as_html(selected_images)}

        Here's all the data you've selected.

        {table}
        """
    )
    return


@app.cell
def _(chart):
    selection = chart.value if len(chart.value) else pd.DataFrame({"digit": [0]})
    return (selection,)


@app.cell
def _(selection):
    most_common_digit = mo.sql(
        f"""
        SELECT digit, COUNT(*) AS frequency
        FROM selection
        GROUP BY digit
        ORDER BY frequency DESC
        LIMIT 1;
        """
    )
    return (most_common_digit,)


@app.cell
def _(most_common_digit, selection):
    _df = mo.sql(
        f"""
        SELECT * FROM selection where digit != {most_common_digit[0, 0]};
        """
    )
    return


@app.cell
def _(chart):
    print("Saving to embedding.html")
    chart.save("embedding.html")
    return


@app.cell
def _(embedding):
    df = dataframe_from_embedding(embedding)
    return (df,)


@app.cell
def _():
    embedding_dimension = 2
    constraint = pymde.Standardized()
    return constraint, embedding_dimension


@app.cell(hide_code=True)
def _(constraint, embedding_dimension, quadratic):
    print("Computing embedding ...")

    with mo.persistent_cache("embedding"):
        embedding, _ = compute_embedding(
            embedding_dimension, constraint, quadratic.value
        )
    return (embedding,)


@app.cell(column=2)
def _():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-q", "--quadratic", action="store_true", default=False)
    try:
        args = parser.parse_args()
    except BaseException:

        class _Namespace: ...

        args = _Namespace()
        args.quadratic = False
    return (args,)


if __name__ == "__main__":
    app.run()
