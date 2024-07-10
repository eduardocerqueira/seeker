#date: 2024-07-10T16:41:46Z
#url: https://api.github.com/gists/cbcd54832c72ee0c854f1375b1995d48
#owner: https://api.github.com/users/Asrst

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

# Generate dummy data
# Generate dummy data with more rows and columns
data = {
    'Category': [chr(65 + i % 26) for i in range(100)],
    'Values': [i % 20 for i in range(100)],
    'Column_1': [i % 10 for i in range(100)],
    'Column_2': [(i % 5) * 2 for i in range(100)],
    'Column_3': [f'Text {i}' for i in range(100)],
    'Column_4': [f'Item {i}' for i in range(100)],
    'Bool_Col': [i % 2 == 0 for i in range(100)]  # Boolean column

}
df = pd.DataFrame(data)

# Create a bar chart
fig = px.bar(df, x='Category', y='Values', color='Bool_Col', 
             barmode='stack', height=400, width=600)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define KPI cards
# Define KPI cards
kpi_card_1 = dbc.Card([
    dbc.CardBody([
        html.H4("KPI Card 1", className="card-title"),
        html.P("Value: 123", className="card-text"),
        dbc.Button("More Info", size="sm", color="primary")
    ])
], className="mb-3", style={'height': '50%', 'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'})

kpi_card_2 = dbc.Card([
    dbc.CardBody([
        html.H4("KPI Card 2", className="card-title"),
        html.P("Value: 456", className="card-text"),
        dbc.Button("More Info", size="sm", color="primary")
    ])
], style={'height': '100%', 'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'})

kpi_card_3 = dbc.Card([
    dbc.CardBody([
        html.H4("KPI Card 3", className="card-title"),
        html.P("Value: 789", className="card-text"),
        dbc.Button("More Info", size="sm", color="primary")
    ])
], style={'height': '100%', 'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)'})

# Define bar graph card
bar_graph_card = dbc.Card([
    dbc.CardBody([
        dcc.Graph(figure=fig)
    ])
], style={'height': '100%', 'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',})

bar_graph_container = dbc.Container([dcc.Graph(figure=fig)],
                    style={'height': '100%', 'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',}
                                    )

# Define the table
table_columns = [{'name': i, 'id': i} for i in df.columns]
table = dash_table.DataTable(
    id='table',
    columns=table_columns,
    data=df.to_dict('records'),
    page_size=10
)

# Define the column selection modal
column_selection_modal = dbc.Modal([
    dbc.ModalHeader("Select Columns to Display"),
    dbc.ModalBody([
        dbc.Checklist(
            options=[{'label': col, 'value': col} for col in df.columns],
            value=[col for col in df.columns],
            id='column-selection-checklist'
        )
    ]),
    dbc.ModalFooter([
        dbc.Button("Close", id='close-column-selection-modal', className='ml-auto')
    ])
], id='column-selection-modal', size='lg')

# Define the table card with three-dot button
table_card = dbc.Card([
    dbc.CardHeader([
        html.Div("Table", className="card-title", style={'display': 'inline-block'}),
        dbc.Button("...", id='open-column-selection-modal', size="sm", color="secondary",
                   style={'float': 'right', 'display': 'inline-block'})
    ]),
    dbc.CardBody([table])
])

# Define the table container with a shadow
table_container = dbc.Container([
    
    html.Div([
        html.Div("Data Table", className="card-title", style={'display': 'inline-block'}),
        dbc.Button("...", id='open-column-selection-modal', size="sm", color="secondary",
                   style={'float': 'right', 'display': 'inline-block'})
    ], className="d-flex justify-content-between align-items-center mb-2"),
    
    dash_table.DataTable(
        id='table',
        columns=[{'name': i, 'id': i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10,
        page_action='native',  # Enables pagination
        style_table={'overflowX': 'auto'}
    )
], style={'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)', 
          'padding': '10px', 
          'borderRadius': '10px'
          })

# Define navigation bar
navbar = dbc.NavbarSimple(
    brand="My Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Define footer
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                "Footer text here",
                className="text-center"
            )
        )
    ),
    fluid=True,
    className="mt-4"
)

# Define the layout
app.layout = dbc.Container([
    navbar,
    dbc.Container([
        # kpi and graph
        dbc.Row([
            dbc.Col([
                kpi_card_1,
                dbc.Row([
                    dbc.Col([kpi_card_2], width=6),
                    dbc.Col([kpi_card_3], width=6)
                ], style={'height': '50%'})
            ], width=6, style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'}),

            dbc.Col([bar_graph_card], width=6)
        ], style={'height': '60vh'}),

        # table
        dbc.Row([
            dbc.Col([table_container], width=12)
        ], style={'marginTop': '20px'}),
        column_selection_modal
    ],),
    footer
], fluid=True,
style={'maxWidth': '1440px'})

# Define the callbacks
@app.callback(
    Output('column-selection-modal', 'is_open'),
    [Input('open-column-selection-modal', 'n_clicks'), Input('close-column-selection-modal', 'n_clicks')],
    [State('column-selection-modal', 'is_open')]
)
def toggle_column_selection_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('table', 'columns'),
    [Input('column-selection-checklist', 'value')]
)
def update_table_columns(selected_columns):
    return [{'name': col, 'id': col} for col in selected_columns]

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
