#date: 2023-01-09T16:52:51Z
#url: https://api.github.com/gists/ed18eafbc0a4938903445b969fdb9f31
#owner: https://api.github.com/users/christopherDT

import numpy as np
import pandas as pd
from onboard.client import OnboardClient
try: # for this you can either create a key.py file with one line: api_key = 'your api key here'
    from key import api_key
except ImportError: # or you can just input your api key when you get the prompt
    api_key = input('Please enter your api_key')

client = OnboardClient(api_key=api_key)

import pandas as pd
from onboard.client.models import PointSelector
from onboard.client.models import TimeseriesQuery, PointData
from onboard.client.dataframes import points_df_from_streaming_timeseries

# query and prepare data for forecasting

# look at data from the last week of July / beginning of August
start_stamp = "2022-07-18T00:00:00Z"
end_stamp = "2022-08-7T00:00:00Z"

# query the server
chws_point_id = ['518876'] # this is the chilled water supply flow (CHWS) from our Laboratory building
timeseries_query = TimeseriesQuery(point_ids=chws_point_id, start=start_stamp , end=end_stamp)
sensor_data = points_df_from_streaming_timeseries(client.stream_point_timeseries(timeseries_query))


# clean data
sensor_data.index = pd.to_datetime(sensor_data.timestamp)
sensor_data.index.name = 'time'
sensor_data

sensor_data = sensor_data.rename(columns={518876 : 'Chilled Water Supply Flow'})

# resample
five_min_data = sensor_data.resample('5T').mean(numeric_only=True) # resample to consistent 5-min interval
five_min_data = five_min_data.interpolate(limit=12)  # allow up to 1-hour of interpolation

# make our feature, operational status (operational = times between 10am-midnight, Monday-Friday)
five_min_data['hour'] = five_min_data.index.hour
five_min_data['weekday'] = five_min_data.index.weekday
five_min_data['workweek'] = five_min_data.apply(lambda x: x.weekday in [0,1,2,3,4], axis=1)
five_min_data['operation'] = five_min_data.apply(lambda x: x.workweek == True and (x.hour >= 10 or x.hour == 0), axis=1)

# now make df_prophet, the data frame we will feed to the model
df_prophet = five_min_data[['Chilled Water Supply Flow', 'operation']].copy()
df_prophet = df_prophet.rename(columns={"Chilled Water Supply Flow": "y"})  # CHWS Flow
df_prophet['ds'] = df_prophet.index
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
df_prophet.head()
