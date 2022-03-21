#date: 2022-03-21T17:11:53Z
#url: https://api.github.com/gists/7755fa9d219734becc00716bd0f2190b
#owner: https://api.github.com/users/kshirsagarsiddharth

app = JupyterDash(__name__)
from datetime import date
app.layout = html.Div([
    dcc.Slider(
        min=df['km_traveled'].min(),
        max=df['km_traveled'].max(),
        value=100.0,
        id='km-travelled-slider'
    ),
    
    dcc.Dropdown(
         id = 'year-picker',
         options = np.sort(df['year'].unique()),
         value = 2017
    ),
    html.Div([
         html.Div(id='km-travelled-slider-output'),
    html.Div(id='year-picker-output')
    ], style={'font-size':'50px'})
   
])

@app.callback(
    Output('km-travelled-slider-output', 'children'),
    Output('year-picker-output', 'children'),
    Input('km-travelled-slider', 'value'),
    Input('year-picker', 'value')
)
def update_output(km_travelled, date_value):
    return f"km_travelled: {km_travelled}", f"date_value: {date_value}"

#app.run_server(mode="inline")
app.run_server()