#date: 2023-01-09T16:52:54Z
#url: https://api.github.com/gists/7e8da19ee68605d355956464d2a46ab9
#owner: https://api.github.com/users/christopherDT

test_days = 3
test_period = test_days * 288 # 5-min intervals per hour * hours in a day

# everything except the last 3 days
df_train = df_prophet.iloc[:(len(df_prophet.index) - test_period)]
# the last 3 days
df_test = df_prophet.iloc[-test_period:]

# create a model
m_future = Prophet(changepoint_prior_scale=0.005,  growth='flat', yearly_seasonality=False)
m_future.add_regressor('operation', mode='multiplicative')

# train
m_future = m_future.fit(df_train)

# predict
future = m_future.predict(df_test)

# postprocess results
future.loc[future['yhat_upper'] < 1, 'yhat_upper'] = 1
future.loc[future['yhat'] < 0, 'yhat'] = 0
future.loc[future['yhat_lower'] < 0, 'yhat_lower'] = 0

fig = m_future.plot(future)
