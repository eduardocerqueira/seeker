#date: 2022-02-22T17:06:37Z
#url: https://api.github.com/gists/6b99b74ef8b0fdf20032965b23e3d1b6
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
from Data.data_getter import data_getter
from GetFixtures import roi_FWD
df = data_getter()
df = roi_FWD

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
main_plot = px.scatter(df, x="roi", y="total_points",
	        size="now_cost", color="position",
            hover_name="web_name")

main_plot.update_layout(
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



app.layout = html.Div(className = 'document', children=[
    navbar,
    html.H1(children = "Welcome to our app", className = "text-center p-3", style = {'color': '#EFE9E7'}),
    html.H3(children = "The home of the best FPL statistics.", className = "text-center p-2 text-light "),
    html.Div(className = "stats-section", id = "stats-section", children = [
            primary_graph
    ]),
])


if __name__ == '__main__':
    app.run_server(debug=True)