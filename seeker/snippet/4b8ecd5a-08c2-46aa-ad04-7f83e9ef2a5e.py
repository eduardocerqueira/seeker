#date: 2022-12-08T17:10:11Z
#url: https://api.github.com/gists/6432f110c8dc6eeb2ea8cb6d62004d82
#owner: https://api.github.com/users/christopherDT

# clean the data
sensor_data.index = pd.to_datetime(sensor_data.timestamp)
sensor_data.index.name = 'time'
five_min_data = sensor_data.resample('5T').mean(numeric_only=True) # resample to consistent 5-min interval
five_min_data = five_min_data.interpolate(limit=12)  # allow up to 1-hour of interpolation

# visualize
names = bldg_df[bldg_df.id.isin(sample_ids)][['id','type']] # get point names for nicer legend display
names_dict = {k:v for (k,v) in zip(names['id'], names['type'])}

five_min_data = five_min_data.rename(columns=names_dict)
five_min_data.plot(figsize=(15, 8))
