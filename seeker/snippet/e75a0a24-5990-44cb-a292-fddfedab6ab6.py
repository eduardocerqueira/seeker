#date: 2023-01-09T16:52:53Z
#url: https://api.github.com/gists/7fe7272b8ec19257248c55ba2e67a1a2
#owner: https://api.github.com/users/christopherDT

from sklearn.metrics import r2_score, mean_squared_error

comp_df = base_forecast[['y','yhat']].copy().rename(columns={'yhat':'base_pred'})
comp_df['regressor_pred'] = regressor_forecast[['yhat']]

print(comp_df.melt(id_vars="y",var_name="comparison_model")\
    .groupby("comparison_model")\
    .apply(lambda x: [round(r2_score(x.y,x.value), 2),round(mean_squared_error(x['y'], x['value']),2)])\
    .reset_index().rename(columns={0:'R-squared, RMSE', 'comparison_model':'model'})
)


comp_df.plot()
