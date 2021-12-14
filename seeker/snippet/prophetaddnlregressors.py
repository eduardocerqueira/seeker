#date: 2021-12-14T17:18:30Z
#url: https://api.github.com/gists/475037e622123b10bc8397b237b639c4
#owner: https://api.github.com/users/paulbruffett

m = NeuralProphet(
    growth='on',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    n_lags=30*2,
    n_forecasts = 30,
    num_hidden_layers=4,
    d_hidden=16,
    learning_rate=0.003,
)
m = m.add_lagged_regressor('WSPM')
m = m.add_lagged_regressor('TEMP')

df_train, df_test = m.split_df(df_a[["ds","y","WSPM","TEMP"]], freq='H', valid_p = 1.0/12)

metrics = m.fit(df_train, freq='H', validation_df=df_test, plot_live_loss=True)