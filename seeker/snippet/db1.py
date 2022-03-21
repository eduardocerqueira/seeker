#date: 2022-03-21T17:09:59Z
#url: https://api.github.com/gists/e8a984fa97aa39805d4cb971ed1c5eb4
#owner: https://api.github.com/users/kshirsagarsiddharth

import plotly.express as px 
import pandas as pd 
from dash import Dash, dcc, html, Input, Output, State
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc 
import numpy as np
df = pd.read_csv('car_price_data.csv').drop('Unnamed: 0', axis = 1)