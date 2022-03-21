#date: 2022-03-21T17:12:32Z
#url: https://api.github.com/gists/6ed556408600fca8e482188689edcdf6
#owner: https://api.github.com/users/kshirsagarsiddharth

from datetime import date
app = JupyterDash(__name__)
app.layout = html.Div([
    dcc.Slider(
        min=df['km_traveled'].min(),
        max=df['km_traveled'].max(),
        value=df['km_traveled'].min(),
        id='km-travelled-slider'
    ),

    dcc.Dropdown(
        id='year-picker',
        options=np.sort(df['year'].unique()),
        value=2017
    ),

    dcc.Slider(
        min=df['tax'].min(),
        max=df['tax'].max(),
        value=df['tax'].min(),
        id='tax-slider'
    ),

    dcc.Slider(
        min=df['engineSize'].min(),
        max=df['engineSize'].max(),
        value=df['engineSize'].min(),
        id='engine-size-slider',
        marks={i: str(i) for i in np.sort(df['engineSize'].unique())}
    ),

    html.Div([
        html.Div(id='km-travelled-slider-output'),
        html.Div(id='year-picker-output'),
        html.Div(id='engine-size-slider-output'),
        html.Div(id='tax-slider-output')
    ], style={'font-size': '50px', 'padding': '20px'})

])

@app.callback(
    Output('km-travelled-slider-output', 'children'),
    Output('year-picker-output', 'children'),
    Output('engine-size-slider-output', 'children'),
    Output('tax-slider-output', 'children'),

    Input('km-travelled-slider', 'value'),
    Input('year-picker', 'value'),
    Input('engine-size-slider', 'value'),
    Input('tax-slider', 'value')
)
def update_output(km_travelled, date_value, engine_size, tax_value):
    return f"km_travelled: {km_travelled}", f"date_value: {date_value}", f"engine_size: {engine_size}", f"Tax: {tax_value}"

#app.run_server(mode="inline")
app.run_server()
Dash app running on http://127.0.0.1:8050/

C:\Users\siddharth_black_pred\anaconda3\envs\geo_env\lib\site-packages\jupyter_dash\jupyter_app.py:139: UserWarning:

The 'environ['werkzeug.server.shutdown']' function is deprecated and will be removed in Werkzeug 2.1.
app = JupyterDash(__name__)
from datetime import date
app.layout = html.Div([
    html.H2('km-travelled'),
    dcc.Slider(
        min=df['km_traveled'].min(),
        max=df['km_traveled'].max(),
        value=100.0,
        id='km-travelled-slider'
    ),
    html.H2('year-picker'),
    dcc.Dropdown(
         id = 'year-picker',
         options = np.sort(df['year'].unique()),
         value = 2017
    ),
    html.H2('tax-picker'),
    dcc.Slider(
        min=df['tax'].min(),
        max=df['tax'].max(),
        value=df['tax'].min(),
        id='tax-slider'
    ),
    html.H2('engine-size'),
     dcc.Slider(
        min=df['engineSize'].min(),
        max=df['engineSize'].max(),
        value=df['engineSize'].min(),
        id='engine-size-slider',
        marks = {i:str(i) for i in np.sort(df['engineSize'].unique())}
    ),
    html.Div([
         html.Div(id='km-travelled-slider-output', children = [html.P('km-travelled-slider-output'),]),
         html.Div(id='year-picker-output'),
         html.Div(id='engine-size-slider-output'),
         html.Div(id='tax-slider-output')
    ], style={'font-size':'50px'})
   
])

@app.callback(
    Output('km-travelled-slider-output', 'children'),
    Output('year-picker-output', 'children'),
    Output('engine-size-slider-output', 'children'),
    Output('tax-slider-output', 'children'),
    
    Input('km-travelled-slider', 'value'),
    Input('year-picker', 'value'),
    Input('engine-size-slider', 'value'),
    Input('tax-slider', 'value')
)
def update_output(km_travelled, date_value,engine_size, tax_value):
    return f"km_travelled: {km_travelled}", f"date_value: {date_value}", f"engine_size: {engine_size}",f"Tax: {tax_value}"

#app.run_server(mode="inline")
app.run_server()
