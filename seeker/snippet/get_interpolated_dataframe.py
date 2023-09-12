#date: 2023-09-12T17:09:23Z
#url: https://api.github.com/gists/131377389eab8099fa37a88ec538515d
#owner: https://api.github.com/users/pipesalas

def get_interpolated_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''Interpolate the dataframe on missing months.
    
    dataf = df.reset_index().copy()
    dataf['time'] = dataf['time'].apply(pd.to_datetime)
    dataf.set_index('time', inplace=True)


    min_month, max_month = dataf.index.min(), dataf.index.max()

    num_periods = (pd.to_datetime(max_month).to_period('M') - pd.to_datetime(min_month).to_period('M')).n
    new_frame = pd.DataFrame({'time': pd.date_range(start=min_month, periods=num_periods+1, freq='M').map(move_date_to_month_beginning)})
    new_frame['time'] = new_frame['time'].apply(pd.to_datetime)
    new_frame.set_index('time', inplace=True)
    new_frame.loc[dataf.index, col] = dataf.loc[dataf.index, col]
    interpolated_frame = new_frame.interpolate(method='cubicspline').copy()
    return interpolated_frame