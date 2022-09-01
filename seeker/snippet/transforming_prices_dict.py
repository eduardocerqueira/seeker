#date: 2022-09-01T16:49:12Z
#url: https://api.github.com/gists/0e966da834349c4903839f4a43f33ac5
#owner: https://api.github.com/users/RocketDav1d

prices_times_quantity_dict = {}
for position in response.results:
    for c_list in prices_dict.values():
        temp_list = []
        for closing_price in c_list:
            total_daily_value_of_instrument = closing_price * position.quantity
            temp_list.append(total_daily_value_of_instrument)

        prices_times_quantity_dict[position.isin] = temp_list
dict_by_index = {}
for instrument in prices_times_quantity_dict.values():
    for index, closing_price in enumerate(instrument):
        temp_list = []
        temp_list.append(closing_price)
        if index in dict_by_index.keys():
            dict_by_index[index] = [*dict_by_index[index], *temp_list]
        else:
            dict_by_index[index] = temp_list
portfolio_values_of_all_instruments = []
for instrument in dict_by_index.values():
    portfolio_values_of_all_instruments.append(sum(instrument))