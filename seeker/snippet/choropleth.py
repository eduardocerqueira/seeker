#date: 2022-07-01T17:10:29Z
#url: https://api.github.com/gists/787e35966bb955522458fe98c7a15cc4
#owner: https://api.github.com/users/okanyenigun

import json
import pandas as pd
import plotly.express as px
from urllib.request import urlopen

#data
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

#figure
fig = px.choropleth(df, geojson = counties, locations='fips', color='unemp',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           scope="usa",
                           labels={'unemp':'unemployment rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()