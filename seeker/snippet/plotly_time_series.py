#date: 2021-09-24T17:13:18Z
#url: https://api.github.com/gists/cd72d21b0635b2607336cd998a88dbac
#owner: https://api.github.com/users/zjwarnes

import plotly.graph_objs as go
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
def zoom(layout, xrange):
    in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
    fig.layout.yaxis.range = [in_view.low.min() - 10, in_view.high.max() + 10]

fig.layout.on_change(zoom, 'xaxis.range')
fig.update_layout(
        template='seaborn',
        margin={'b': 15}, hovermode='x', autosize=True,
        title={'text': 'Bitcoin Price', 'x': 0.5},
        yaxis_title ='Price',
        xaxis={
            'range': [df.index.min(), df.index.max()], 'rangeslider_visible': True,
            'rangebreaks':[dict(values=dt_breaks)], 
            'rangeselector': dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            )
        }
)
fig