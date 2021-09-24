#date: 2021-09-24T17:13:18Z
#url: https://api.github.com/gists/cd72d21b0635b2607336cd998a88dbac
#owner: https://api.github.com/users/zjwarnes

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )
)
fig.update_layout(
    title='Bitcoin Price',
    title_x=0.5,
    yaxis_title ='Price'
)
fig.show()