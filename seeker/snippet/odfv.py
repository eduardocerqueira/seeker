#date: 2021-09-21T16:59:12Z
#url: https://api.github.com/gists/4dd5bb783e2a03ad1d9e3f65941733c5
#owner: https://api.github.com/users/adchia

# Define a request data source which encodes features / information only 
# available at request time (e.g. part of the user initiated HTTP request)
input_request = RequestDataSource(
    name="vals_to_add",
    schema={
        "val_to_add": ValueType.INT64,
        "val_to_add_2": ValueType.INT64
    }
)

# Use the input data and feature view features to create new features
@on_demand_feature_view(
   inputs={
       'driver_hourly_stats': driver_hourly_stats_view,
       'vals_to_add': input_request
   },
   features=[
     Feature(name='conv_rate_plus_val1', dtype=ValueType.DOUBLE),
     Feature(name='conv_rate_plus_val2', dtype=ValueType.DOUBLE)
   ]
)
def transformed_conv_rate(features_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df['conv_rate_plus_val1'] = (features_df['conv_rate'] + features_df['val_to_add'])
    df['conv_rate_plus_val2'] = (features_df['conv_rate'] + features_df['val_to_add_2'])
    return df