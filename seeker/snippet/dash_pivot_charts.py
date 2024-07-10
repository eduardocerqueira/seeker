#date: 2024-07-10T16:40:28Z
#url: https://api.github.com/gists/2e52eefc84d9a48ae64b3ca83c4c1a09
#owner: https://api.github.com/users/Asrst

import dash
from dash.dependencies import Input, Output
from dash import html
import dash_pivottable
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'My Dash example'


data = data = [
["Total Bill","Tip","Payer Gender","Payer Smoker","Day of Week","Meal","Party Size"],
[16.99,1.01,"Female","Non-Smoker","Sunday","Dinner",2],
[10.34,1.66,"Male","Non-Smoker","Sunday","Dinner",3],
[21.01,3.5,"Male","Non-Smoker","Sunday","Dinner",3],
[23.68,3.31,"Male","Non-Smoker","Sunday","Dinner",2],
[24.59,3.61,"Female","Non-Smoker","Sunday","Dinner",4],
[14.78,3.23,"Male","Smoker","Sunday","Dinner",2],
[10.27,1.71,"Male","Smoker","Sunday","Dinner",2],
[35.26,5,"Female","Non-Smoker","Sunday","Dinner",4],
[15.42,1.57,"Male","Non-Smoker","Sunday","Dinner",2],
[20.65,3.35,"Male","Non-Smoker","Saturday","Dinner",3],
[17.92,4.08,"Male","Non-Smoker","Saturday","Dinner",2],
[20.29,2.75,"Female","Non-Smoker","Saturday","Dinner",2],
[15.77,2.23,"Female","Non-Smoker","Saturday","Dinner",2],
[13,2,"Female","Smoker","Thursday","Lunch",2],
[16.4,2.5,"Female","Smoker","Thursday","Lunch",2],
[20.53,4,"Male","Smoker","Thursday","Lunch",4],
[16.47,3.23,"Female","Smoker","Thursday","Lunch",3],
[26.59,3.41,"Male","Smoker","Saturday","Dinner",3],
[38.73,3,"Male","Smoker","Saturday","Dinner",4],
[24.27,2.03,"Male","Smoker","Saturday","Dinner",2],
[12.76,2.23,"Female","Smoker","Saturday","Dinner",2],
[30.06,2,"Male","Smoker","Saturday","Dinner",3],
[12.9,1.1,"Female","Smoker","Saturday","Dinner",2],
[28.15,3,"Male","Smoker","Saturday","Dinner",5],
[11.59,1.5,"Male","Smoker","Saturday","Dinner",2],
[7.74,1.44,"Male","Smoker","Saturday","Dinner",2],
[30.14,3.09,"Female","Smoker","Saturday","Dinner",4],]

app.layout = dbc.Container([
    
        dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("About", href="#")),
        ],
        brand="My Dash App",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    html.Br(),
    dash_pivottable.PivotTable(
        id='table',
        data=data,
        cols=['Day of Week'],
        rows=['Party Size'],
        colOrder="key_a_to_z",
        rowOrder="key_a_to_z",
        rendererName="Grouped Column Chart",
        aggregatorName="Average",
        vals=["Total Bill"],
        valueFilter={'Day of Week': {'Thursday': False}}
    ),

    html.Div(
        id='output'
    )
],  
fluid=True,
style={'maxWidth': '1400px', 'margin': 'auto'}
)


@app.callback(Output('output', 'children'),
              [Input('table', 'cols'),
               Input('table', 'rows'),
               Input('table', 'rowOrder'),
               Input('table', 'colOrder'),
               Input('table', 'aggregatorName'),
               Input('table', 'rendererName')])
def display_props(cols, rows, row_order, col_order, aggregator, renderer):
    return [
        # html.P(str(cols), id='columns'),
        # html.P(str(rows), id='rows'),
        html.P(str(row_order), id='row_order'),
        html.P(str(col_order), id='col_order'),
        # html.P(str(aggregator), id='aggregator'),
        # html.P(str(renderer), id='renderer'),
    ]


if __name__ == '__main__':
    app.run_server(debug=True)