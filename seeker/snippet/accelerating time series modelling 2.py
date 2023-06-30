#date: 2023-06-30T16:37:05Z
#url: https://api.github.com/gists/bca57ee85895938aca5520893194c381
#owner: https://api.github.com/users/davidemastricci

air_pass['Month'] = pd.to_datetime(air_pass.Month)
air_pass = air_pass.set_index(air_pass.Month)
air_pass.drop(columns=['Month'], inplace = True)
print('Column datatypes= \n',air_pass.dtypes)
air_pass