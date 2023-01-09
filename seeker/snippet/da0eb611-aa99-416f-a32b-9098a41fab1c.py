#date: 2023-01-09T16:52:56Z
#url: https://api.github.com/gists/1c41e77347b3f818de4fb67e23dbc6cd
#owner: https://api.github.com/users/christopherDT

# resample from original sensor data we queried earlier
hourly_data = sensor_data.resample('60T').mean(numeric_only=True) # resample to consistent 1 hour intervals

# make our feature, operational status (operational = times between 10am-midnight, Monday-Friday)
hourly_data['hour'] = hourly_data.index.hour
hourly_data['weekday'] = hourly_data.index.weekday
hourly_data['workweek'] = hourly_data.apply(lambda x: x.weekday in [0,1,2,3,4], axis=1)
hourly_data['operation'] = hourly_data.apply(lambda x: x.workweek == True and (x.hour >= 10 or x.hour == 0), axis=1)

# now make the data frame we will feed to the model
df_prophet_hourly = hourly_data[['Chilled Water Supply Flow', 'operation']].copy()
df_prophet_hourly = df_prophet_hourly.rename(columns={"Chilled Water Supply Flow": "y"})  # CHWS Flow
df_prophet_hourly['ds'] = df_prophet_hourly.index
df_prophet_hourly['ds'] = df_prophet_hourly['ds'].dt.tz_localize(None)
df_prophet_hourly.head()

# make a model for the hourly data
hourly_model = Prophet(changepoint_prior_scale=0.005, yearly_seasonality=False)
hourly_model.add_regressor('operation', mode='multiplicative')
hourly_model = hourly_model.fit(df_prophet_hourly)

hourly_forecast = hourly_model.predict(df_prophet_hourly)
hourly_forecast['y'] = df_prophet_hourly['y'].reset_index(drop = True)

# train/test split

test_days = 3
test_period = test_days * 24 # 24 hours per day

# everything except the last 3 days
df_train_hourly = df_prophet_hourly.iloc[:(len(df_prophet_hourly.index) - test_period)]
# the last 3 days
df_test_hourly = df_prophet_hourly.iloc[-test_period:]

# create a model
m_future_hourly = Prophet(changepoint_prior_scale=0.005,  growth='flat', yearly_seasonality=False)
m_future_hourly.add_regressor('operation', mode='multiplicative')

# train
m_future_hourly = m_future_hourly.fit(df_train_hourly)

# predict
future_hourly = m_future_hourly.predict(df_test_hourly)

# postprocess results
future_hourly.loc[future_hourly['yhat_upper'] < 1, 'yhat_upper'] = 1
future_hourly.loc[future_hourly['yhat'] < 0, 'yhat'] = 0
future_hourly.loc[future_hourly['yhat_lower'] < 0, 'yhat_lower'] = 0

# forecast accuracy
future_hourly.index = future_hourly.ds
future_hourly['y'] = df_test_hourly['y'].values

print('')
print(f'R-squared = {round(r2_score(future_hourly["y"], future_hourly["yhat"]), 2)}')
print(f'mean squared error = {round(mean_squared_error(future_hourly["y"], future_hourly["yhat"]),2)}')

future_hourly[['y', 'yhat']].plot(figsize=(8,5))
