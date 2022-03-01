#date: 2022-03-01T17:12:04Z
#url: https://api.github.com/gists/d99cca7f45ebf38cf83c66adc58d6166
#owner: https://api.github.com/users/jasher4994

#Package import
import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import requests

#Function and data import
from GetFixtures import FWD_roi, MID_roi, DEF_roi, GK_roi, EG, ECS
df = FWD_roi

#initialising app
app = dash.Dash(
    external_stylesheets = [dbc.themes.DARKLY])


#colours
colors = {
    'black' : '#1A1B25',
    'red' : '#F8C271E',
    'white' : '#EFE9E7',
    'background' : '#333333',
    'text' : '#FFFFFF'
}


#PLOTS
main_plot = px.scatter(df, x="ROI", y="EG(6)",
	        size="Cost", color="Team",
            hover_name="Name", 
            title = "test")


main_plot.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)


EG_plot = px.bar(EG, x="team", y=["EG - one week", "EG - three week", "EG - six week" ],
            color_discrete_sequence = px.colors.qualitative.Safe, 
            hover_name="team")

EG_plot.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)


ECS_plot = px.bar(ECS, x="team", y=["ECS - one week", "ECS - three week", "ECS - six week" ],
            color_discrete_sequence = px.colors.qualitative.Safe, 
            hover_name="team")


ECS_plot.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

#COMPONENTS

#navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("How it works", href="#"),
                dbc.DropdownMenuItem("The statistics", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="Explore",
        ),
    ],
    brand="FPL dashboard",
    brand_href="#",
    color="dark",
    dark=True,
)

#Primary graph
primary_graph = dcc.Graph(
            className = 'primary-graph',
            id='primary-graph',
            figure= main_plot,
            style = {'backgroundColor' : colors['background']}
        )

EG_graph = dcc.Graph(
            className = 'EG-graph',
            id='EG-graph',
            figure= EG_plot,
            style = {'backgroundColor' : colors['background']}
)

ECS_graph = dcc.Graph(
            className = 'ECS-graph',
            id='ECS-graph',
            figure= ECS_plot,
            style = {'backgroundColor' : colors['background']}
)

#Primary graph
FWD_table = dbc.Table.from_dataframe(FWD_roi, hover= True, color="secondary", size = "sm", bordered=True) 
MID_table = dbc.Table.from_dataframe(MID_roi, hover= True, color="secondary", size = "sm", bordered=True) 
DEF_table = dbc.Table.from_dataframe(DEF_roi, hover= True, color="secondary", size = "sm", bordered=True) 
GK_table = dbc.Table.from_dataframe(GK_roi, hover= True, color="secondary", size = "sm", bordered=True) 


################
#### LAYOUT ####
################

app.layout = html.Div(className = 'document', children=[
    navbar,
    html.H1(children = "Welcome to our app", className = "text-center p-2", style = {'color': '#EFE9E7'}),
    html.H3(children = "The home of the best FPL statistics.", className = "text-center p-1 text-light "),
    html.Div(className = "stats-section ", id = "stats-section", children = [
            primary_graph,
            EG_graph,
            ECS_graph,
            html.Div(className = "flex-container", children = [
                dbc.Row(
                    [
                        dbc.Col(html.Div(FWD_table, className = "p-3")),
                        dbc.Col(html.Div(MID_table, className = "p-3")),
                        dbc.Col(html.Div(DEF_table, className = "p-3")),
                        dbc.Col(html.Div(GK_table, className = "p-3"))
                    ]
                )
            ])
            
            
    ]),
])


if __name__ == '__main__':
    app.run_server(debug=True)