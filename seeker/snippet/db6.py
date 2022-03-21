#date: 2022-03-21T17:15:33Z
#url: https://api.github.com/gists/9490810d914062c41588011a9c8dc855
#owner: https://api.github.com/users/kshirsagarsiddharth

app = JupyterDash(__name__)
from datetime import date
app.layout = html.Div([
    html.H2('km-travelled'),
    dcc.Slider(
        min=df['km_traveled'].min(),
        max=df['km_traveled'].max(),
        value=100.0,
        id='km-travelled-slider',
        tooltip={"placement": "bottom"}
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
        id='tax-slider',
        tooltip={"placement": "bottom"}
    ),
    html.H2('engine-size'),
     dcc.Slider(
        min=df['engineSize'].min(),
        max=df['engineSize'].max(),
        value=df['engineSize'].min(),
        id='engine-size-slider',
        marks = {i:str(i) for i in np.sort(df['engineSize'].unique())},
        tooltip={"placement": "bottom"}
    ),
    
    html.H2('km-per-liter'),
     dcc.Slider(
        min=df['km_per_liters'].min(),
        max=df['km_per_liters'].max(),
        value=df['km_per_liters'].min(),
        id='km-per-liters-slider',
        tooltip={"placement": "bottom"}
    ),
    
    html.H2('model-dropdown'),
    dcc.Dropdown(
         id = 'model-dropdown',
         options = np.sort(df['model'].unique()),
         value = df['model'][0]
    ),
    html.H2('transmission-dropdown'),
    dcc.Dropdown(
         id = 'transmission-dropdown',
         options = np.sort(df['transmission'].unique()),
         value = df['transmission'][0]
    ),
    
    html.H2('fuel-type-dropdown'),
    dcc.Dropdown(
         id = 'fuel-type-dropdown',
         options = np.sort(df['fuel_type'].unique()),
         value = df['fuel_type'][0]
    ),
   html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    
    
    html.Div([
         html.Div(id='km-travelled-slider-output', children = [html.P('km-travelled-slider-output'),]),
         html.Div(id='year-picker-output'),
         html.Div(id='engine-size-slider-output'),
         html.Div(id='tax-slider-output'),
         html.Div(id='km-per-liters-slider-output'),
         html.Div(id='model-dropdown-output'),
         html.Div(id='transmission-dropdown-output'),
         html.Div(id='fuel-type-dropdown-output')
    ], style={'font-size':'20px'})
   
])

@app.callback(
    Output('km-travelled-slider-output', 'children'),
    Output('year-picker-output', 'children'),
    Output('engine-size-slider-output', 'children'),
    Output('tax-slider-output', 'children'),
    Output('km-per-liters-slider-output', 'children'),
    Output('model-dropdown-output', 'children'),
    Output('transmission-dropdown-output', 'children'),
    Output('fuel-type-dropdown-output', 'children'),
    
    Input('submit-button-state', 'n_clicks'),
    State('km-travelled-slider', 'value'),
    State('year-picker', 'value'),
    State('engine-size-slider', 'value'),
    State('tax-slider', 'value'),
    State('km-per-liters-slider', 'value'),
    State('model-dropdown', 'value'),
    State('transmission-dropdown', 'value'),
    State('fuel-type-dropdown', 'value')
)
def update_output(n_clicks,km_travelled, date_value,engine_size, tax_value, km_per_liters, model_name, transmission, fuel_type):
    return (f"km_travelled: {km_travelled}", 
            f"date_value: {date_value}", 
            f"engine_size: {engine_size}",
            f"Tax: {tax_value}",
            f"Km Per Liters: {km_per_liters}", 
            f"model name: {model_name}", 
            f"transmission: {transmission}", 
            f"fuel_type: {fuel_type}")

#app.run_server(mode="inline")
app.run_server()
