#date: 2025-03-05T16:45:09Z
#url: https://api.github.com/gists/221c870bd1581f5065c067962918b8cc
#owner: https://api.github.com/users/Mo-Gul

import marimo

__generated_with = "0.11.14"
app = marimo.App(app_title="spikes")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Identifying Spikes

        Suppose you have some timeseries data with occasional spikes in it and you want to identify them.
        This is what I came up with as a solution using Pandas and [median absolute deviation](http://en.wikipedia.org/wiki/Median_absolute_deviation).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    return go, mo, np, pd, px


@app.cell
def _(pd):
    # following option will be the default in pandas v3.0
    # it is recommended to turn it on in the manual
    # REF: <https://pandas.pydata.org/docs/user_guide/copy_on_write.html>
    pd.options.mode.copy_on_write = True
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Make a fake time series that could represent something like temperature.
        It can change in either direction but you don't expect big jumps in a single step so the cumulative sum of random numbers makes a decent test case.
        """
    )
    return


@app.cell
def _(np, pd):
    rng = np.random.default_rng()
    ts = pd.Series(rng.standard_normal(500), index=pd.date_range("1/1/2000", periods=500))
    ts = ts.cumsum()
    return rng, ts


@app.cell
def _(mo):
    mo.md(r"Now I'll make 20 random spikes to put in.")
    return


@app.cell
def _(rng):
    spikes = 25 * rng.random(20) - 12
    return (spikes,)


@app.cell
def _(mo):
    mo.md(r"...and choose 20 random locations to put the spike.")
    return


@app.cell
def _(rng, spikes, ts):
    spike_points = rng.integers(0, 500, 20)
    ts.iloc[spike_points] = ts.iloc[spike_points] + spikes
    return (spike_points,)


@app.cell
def _(mo):
    mo.md(r"Here are the fake data with red vertical lines marking where the spikes are.")
    return


@app.cell
def _(px, spike_points, ts):
    fig = px.line(ts)
    for spike_point in spike_points:
        fig.add_vline(x=ts.index[spike_point], line_color="red", layer="below")
    fig
    return fig, spike_point


@app.cell
def _(mo):
    mo.md(
        r"""
        Some of the spikes are very small ... too small to really be detected.
        That's fine though.
        Here are the basic stats on the magnitude of the fake spikes:
        """
    )
    return


@app.cell
def _(pd, spikes):
    # give a non-integer column name to avoid
    # """
    # UserWarning:
    # DataFrame has integer column names.
    # This is not supported and can lead to bugs.
    # Please use strings for column names.
    # """
    pd.DataFrame(abs(spikes), columns=["0"]).describe()
    return


@app.cell
def _(mo):
    mo.md(r"Put this stuff into a pandas dataframe to keep track of it:")
    return


@app.cell
def _(pd, ts):
    df = pd.DataFrame(ts, columns=["Data"])
    return (df,)


@app.cell
def _(df, np, spike_points, spikes):
    df["Spike"] = False
    df.loc[df.index[spike_points], "Spike"] = True
    df["SpikeVal"] = np.nan
    df.loc[df.index[spike_points], "SpikeVal"] = spikes
    return


@app.cell
def _(mo):
    mo.md(r"Here's a bit of the dataframe to see what it looks like (these rows are all spikes)")
    return


@app.cell
def _(df):
    df[df["Spike"]].head()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Detecting Spikes

        This is where I'm actually going to try to detect the spikes.
        I'm going to use [MAD](http://en.wikipedia.org/wiki/Median_absolute_deviation) with a rolling window.
        I'm going to look at each element of the data within the context of window of 3 items.
        I'm measure the difference between that item and the median of all three items.
        Then I'm going to subtract MAD of all 3 items in the window from that measure. So, as I see it, my $Relative Madness$ measure should tell me how far off from it's neighbors each value is.
        """
    )
    return


@app.cell
def _(np):
    def relative_madness(x):
        return abs(x[1] - np.median(x)) - np.median(abs(x - np.median(x)))
    return (relative_madness,)


@app.cell
def _(df, relative_madness):
    # The commented code below throughs
    # """
    # FutureWarning:
    # Series.__getitem__ treating keys as positions is deprecated.
    # In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior).
    # To access a value by position, use `ser.iloc[pos]`
    # """
    # To avoid this only give the *values*.
    # df["Madness"] = df["Data"].rolling(3, center=True).apply(relative_madness)
    df["Madness"] = df["Data"].rolling(3, center=True).apply(lambda s: relative_madness(s.values))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        This boxplot shows me that the known spiked values do tend to have higher $Relative Madness$ values than the non-spiked values.
        Keep in mind that many of the spikes are tiny and won't be detectable.
        It the case that I'm looking at right now (remember the numbers change every time this code is run because it's randomly generated), I could chose a $Madness$ threshold of $2$ and it would not give me any false positives and would detect around $75\%$ of the spikes.
        """
    )
    return


@app.cell
def _(df, px):
    _fig2 = px.box(df, x="Spike", y="Madness")
    _fig2
    return


@app.cell
def _(mo):
    mo.md(r"Label values with a $Madness > 2$ as being detected spikes.")
    return


@app.cell
def _(df):
    df["SpikeFound"] = False
    df.loc[df["Madness"] > 2, "SpikeFound"] = True
    return


@app.cell
def _(mo):
    mo.md(r"Plot the results")
    return


@app.cell
def _(df, fig, go, np):
    _detected = df[df["SpikeFound"]]

    _fig3 = go.Figure(fig)
    _fig3.add_trace(go.Scatter(
        x=df.index,
        y=df["Madness"],
        name="Madness",
        mode="lines",
    ))
    _fig3.add_trace(go.Scatter(
        x=_detected.index,
        y=np.repeat(10, _detected["SpikeFound"].sum()),
        mode="markers",
    ))
    _fig3
    return


@app.cell
def _(mo):
    mo.md(r"Here's how the prediction did in terms of false positives and false negatives:")
    return


@app.cell
def _(df, pd):
    pd.crosstab(df["Spike"], df["SpikeFound"])
    return


@app.cell
def _(df, px):
    _spikedf = df[df["Spike"]]
    _fig4 = px.scatter(
        _spikedf,
        x=abs(_spikedf["SpikeVal"]),
        y="Madness",
        color="SpikeFound",
        labels=dict(x="SpikeVal"),
    )
    _fig4
    return


if __name__ == "__main__":
    app.run()
