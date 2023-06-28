#date: 2023-06-28T17:11:00Z
#url: https://api.github.com/gists/24f59909b39cf1affbf69d7c0dbc377c
#owner: https://api.github.com/users/olin000


import gpxpy
import matplotlib.pyplot as plt
import datetime
from geopy import distance
from math import sqrt, floor
import numpy as np
import pandas as pd
import haversine

import os
from os import listdir
from os.path import isfile, join

# All the sections you PB's for in meters:
sections = [1000,(1000*1.60934),3000,5000,(5000*1.60934),10000,15000,(10000*1.60934),20000,21097.5,25000,30000,40000,42195]

path = os.path.join(os.path.abspath(''), 'tracks')
allfiles = [f for f in listdir(path) if isfile(join(path, f))]

df_final = pd.DataFrame(columns=['time', 'distance', 'minutes_per_kilometer'])

for file in allfiles:
  
  path = os.path.join(os.path.abspath(''), 'tracks', file)

  gpx_file = open(path, 'r')
  gpx = gpxpy.parse(gpx_file)

  df = pd.DataFrame(columns=['lon', 'lat', 'alt', 'time'])
  
  for segment in gpx.tracks[0].segments: # all segments

    data = segment.points

    for point in data:
        df = df.append({'lon': point.longitude, 'lat' : point.latitude, 'alt' : point.elevation, 'time' : point.time}, ignore_index=True)
    
  df = df.sort_values(by=['time'])
  df = df.reset_index()

  df = df.fillna(method='ffill')
  df = df.fillna(method='bfill')
  
  # Create a column with values that are 'shifted' one forwards, so we can create calculations for differences.
  df['lon-start'] = df['lon']
  df['lon-start'].iloc[-1] = np.nan
  df['lon-start'] = np.roll(df['lon-start'], 1)
  df['lat-start'] = df['lat']
  df['lat-start'].iloc[-1] = np.nan
  df['lat-start'] = np.roll(df['lat-start'], 1)
  df['alt-start'] = df['alt']
  df['alt-start'].iloc[-1] = np.nan
  df['alt-start'] = np.roll(df['alt-start'], 1)
  df['time-start'] = df['time']
  df['time-start'].iloc[-1] = np.nan
  df['time-start'] = np.roll(df['time-start'], 1)
  df = df.fillna(method='bfill')
  
  df['time'] = pd.to_datetime(df['time'], utc=True)
  df['time'] = df['time'].dt.tz_localize(tz=None)
  df['time-start'] = pd.to_datetime(df['time-start'], utc=True)
  df['time-start'] = df['time-start'].dt.tz_localize(tz=None)
  
  df['distance_dis_2d'] = df.apply(lambda x: distance.distance((x['lat-start'], x['lon-start']), (x['lat'], x['lon'])).m, axis = 1)
  df['alt_dif'] = df.apply(lambda x: x['alt-start'] - x['alt'], axis=1)
  df['distance_dis_3d'] = df.apply(lambda x: sqrt(x['distance_dis_2d']**2 + (x['alt_dif'])**2), axis=1)
  df['time_delta'] = df.apply(lambda x: (x['time'] - x['time-start']).total_seconds(), axis=1)
  
  df_selected = df.loc[:, ['distance_dis_3d','time_delta']]

  df_selected['distance_cumsum'] = df_selected['distance_dis_3d'].cumsum()
  df_selected['time_cumsum'] = df_selected['time_delta'].cumsum()
  
  # Here we loop over sections
  for section in sections:

      if df['distance_dis_3d'].sum() < section: # If the total distance of the workout is smaller then the section we're looking for we can skip this iteration.
          continue

      df_output = pd.DataFrame(columns=['date', 'section', 'filename', 'time', 'distance', 'minutes_per_kilometer', 'total_distance', 'total_time'])

      for i in range(len(df_selected.index)):

          df_section = df_selected[(df_selected['distance_cumsum'] - df_selected['distance_cumsum'].iat[i]) >= section]
          if(len(df_section.index) != 0):
              time = df_section['time_cumsum'].iat[0] - df_selected['time_cumsum'].iat[i]
              distance_i = df_section['distance_cumsum'].iat[0] - df_selected['distance_cumsum'].iat[i]
              minutes_per_kilometer = (time/60)/(distance_i/1000)
              df_output = df_output.append({'date': df['time'].min(), 'section': section, 'filename': file, 'time': time, 'distance': distance_i, 'minutes_per_kilometer': minutes_per_kilometer, 'total_distance': df['distance_dis_3d'].sum(), 'total_time': df['time_delta'].sum()}, ignore_index=True)

      s_best = df_output.loc[df_output['minutes_per_kilometer'].idxmin()]
      df_final = df_final.append(s_best)

df_final['start_index_best_section'] = df_final.index
df_final = df_final.set_index(['filename','section'])