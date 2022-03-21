#date: 2022-03-21T17:11:07Z
#url: https://api.github.com/gists/a858be858ac60a58e384660e952cab3e
#owner: https://api.github.com/users/kshirsagarsiddharth

app = JupyterDash(__name__)

app.layout = html.Div([
    dcc.Slider(
        min = df['km_traveled'].min(),
        max = df['km_traveled'].max(),
        value = 100.0,
        id = 'km-travelled-slider'
    ),
    html.Div(id = 'slider-output-container')
])
@app.callback(
    Output('slider-output-container','children'),
    Input('km-travelled-slider','value')
)
def update_output(value):
    return 'You have selected "{}"'.format(value)

#app.run_server(mode = "inline")
app.run_server()
#Dash app running on http://127.0.0.1:8050/