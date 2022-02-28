#date: 2022-02-28T17:13:14Z
#url: https://api.github.com/gists/48b7963315f889d06b96068a98dca2f2
#owner: https://api.github.com/users/discarn8

import plotly.graph_objects as go
import dash
from dash import dcc, html
import requests
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
app.layout = html.Div([
        #Display the graph - specify the callback to call on
        dcc.Graph(id='live-update-graph'),
        #Define the interval - here it is set to update every 10 seconds
        dcc.Interval(
            id='interval-component',
            interval=1*10000, # in milliseconds
            n_intervals=0
        )
    ]
)

#Callback - defined by name and interval
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))

#Specify the steps of our function to be processed when called
def update_graph_live(n):
    room1 =requests.get('http://192.168.0.11/temp.txt', timeout=None).text
    room2 =requests.get('http://192.168.0.12/temp.txt', timeout=None).text
    room3 =requests.get('http://192.168.0.13/temp.txt', timeout=None).text
    room4 =requests.get('http://192.168.0.14/temp.txt', timeout=None).text

    fig = go.Figure(data=go.Heatmap(
                                #Input our data fields to z
                                z=[[room1, room2],[room3, room4]],
                                #Assign text labels and the value to each heatmap section
                                text=[["ROOM1<br>"+room1+"&deg;", "ROOM2<br>"+room2+"&deg;"],
                                  ["ROOM3<br>"+room3+"&deg;", "ROOM$<br>"+room4+"&deg;"]
                                 ],
                                #This simply adds a visibile gap between the heatmap squares
                                #Remove it if it is unecessary
                                xgap=5,
                                ygap=5,
                                #How do we want our text to appear?
                                texttemplate="%{text}",
                                textfont={"size":50,"family":"Courier New"},
                                colorscale='RdYlBu', 
                                #Reverse the colors of our scale?
                                reversescale=True, 
                                #How do we want our colorscale to appear?
                                colorbar=dict(thickness=20,
                                            ticklen=3, 
                                            tickcolor='orange',
                                            tickfont=dict(size=14, 
                                                        color='orange'
                                                         ),
                                              orientation="h"
                                             )
                            )
               )

    #Manualy define the upper and lower range for our colorscale, regardless of data values
    fig.data[0].update(zmin=60, zmax=90)
    #Remove the x and y axis ticks and make the background black
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False,
                    xaxis_visible=False, xaxis_showticklabels=False,
                    paper_bgcolor='black'
                    )
    #Return our formatted figure
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8093)