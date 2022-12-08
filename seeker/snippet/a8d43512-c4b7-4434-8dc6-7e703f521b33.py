#date: 2022-12-08T17:10:10Z
#url: https://api.github.com/gists/25beccbae2c286814fc5f22ddaa01b25
#owner: https://api.github.com/users/christopherDT

from onboard.client.models import TimeseriesQuery, PointData
from onboard.client.dataframes import points_df_from_streaming_timeseries

# look at data from the last week of July / beginning of August
start_stamp = "2022-07-18T00:00:00Z"
end_stamp = "2022-08-7T00:00:00Z"

# query the server
timeseries_query = TimeseriesQuery(point_ids=sample_ids, start=start_stamp , end=end_stamp)
sensor_data = points_df_from_streaming_timeseries(client.stream_point_timeseries(timeseries_query))
sensor_data.head() # always good to sneak a peak at what the data looks like
