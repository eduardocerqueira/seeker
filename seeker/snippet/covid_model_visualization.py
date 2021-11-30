#date: 2021-11-30T17:03:05Z
#url: https://api.github.com/gists/c17aaab5207bdb02f3c1cb2e999143fd
#owner: https://api.github.com/users/basselkarami

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from covid_model import *

def agent_portrayal(agent):
    portrayal = {
        'Shape': 'circle',
        'Layer': 0,
        'r': 1,
        'Color': 'lightblue'}

    # (Un)masked agents show up as (non-)filled circles
    if agent.masked == True:
        portrayal['Filled'] = 'true'

    if agent.infected == True:
        portrayal['Color'] = 'red'

    if agent.immune == True:
        portrayal['Color'] = 'green'

    return portrayal

grid = CanvasGrid(agent_portrayal, 50, 50, 500, 500)

line_charts = ChartModule([
    {'Label': 'Susceptible', 'Color': 'lightblue'}, 
    {'Label': 'Infected', 'Color': 'red'},
    {'Label': 'Recovered & Immune', 'Color': 'green'}])

server = ModularServer(CovidModel,
                       [grid, line_charts],
                       'COVID Simulation Model',
                       model_params)

server.port = 8521  # default port if unspecified
server.launch()