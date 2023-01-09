#date: 2023-01-09T16:52:52Z
#url: https://api.github.com/gists/7b7a06265acb654fd880996e52b7b5d6
#owner: https://api.github.com/users/christopherDT

# !pip install prophet==1.0 # install prophet if you haven't already
from prophet import Prophet
import warnings
warnings.simplefilter("ignore", FutureWarning) # this is just because prophet uses pd.append() instead of pd.concat()

# create a model without our regressor
base_model = Prophet(changepoint_prior_scale=0.005, yearly_seasonality=False)
base_model = base_model.fit(df_prophet)

base_forecast = base_model.predict(df_prophet)
base_forecast['y'] = df_prophet['y'].reset_index(drop = True)

# create a model with the regressor
regressor_model = Prophet(changepoint_prior_scale=0.005, yearly_seasonality=False)
regressor_model.add_regressor('operation', mode='multiplicative')
regressor_model = regressor_model.fit(df_prophet)

regressor_forecast = regressor_model.predict(df_prophet)
regressor_forecast['y'] = df_prophet['y'].reset_index(drop = True)
