#date: 2021-09-24T17:13:18Z
#url: https://api.github.com/gists/cd72d21b0635b2607336cd998a88dbac
#owner: https://api.github.com/users/zjwarnes

def zoom(layout, xrange):
    in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
    fig.layout.yaxis.range = [in_view.low.min() - 10, in_view.high.max() + 10]

fig.layout.on_change(zoom, 'xaxis.range')