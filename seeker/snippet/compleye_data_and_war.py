#date: 2022-04-15T17:06:16Z
#url: https://api.github.com/gists/6c0f0264e630e90c59ae88f86bde3257
#owner: https://api.github.com/users/victorfrutuoso

complete_data_and_war = pd.merge(complete_data_2, war_values_merged, how = 'left', on=['full_name','player_id','year'])