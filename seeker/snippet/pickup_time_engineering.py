#date: 2022-02-28T16:57:36Z
#url: https://api.github.com/gists/bde9db81a10375707920b9a1668212b8
#owner: https://api.github.com/users/JakobLS

# Pickup hour
nyc_trips["pickup_hour"] = nyc_trips["pickup_time"].dt.hour

# Pickup day in the week (Monday = 0)
nyc_trips["day_of_week"] = nyc_trips["pickup_time"].dt.dayofweek

# Pickup month
nyc_trips["pickup_month"] = nyc_trips["pickup_time"].dt.month

# Pickup year
nyc_trips["pickup_year"] = nyc_trips["pickup_time"].dt.year

# Trip duration, in minutes
nyc_trips["trip_duration"] = np.round(((nyc_trips["dropoff_time"] - nyc_trips["pickup_time"]).dt.total_seconds() / 60), 0)