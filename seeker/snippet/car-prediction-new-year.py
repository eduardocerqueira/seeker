#date: 2022-01-12T17:09:16Z
#url: https://api.github.com/gists/3786204d9d6efda3b1b8b80543b048e9
#owner: https://api.github.com/users/suhasmaddali

data['Present Year'] = 2021
data['Years Of Manufacture'] = data['Present Year'] - data['Year']
data.drop(['Present Year'], inplace = True, axis = 1)