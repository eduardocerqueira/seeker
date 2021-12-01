#date: 2021-12-01T17:08:05Z
#url: https://api.github.com/gists/f78ad5407b6534b2cb30677c353a8e8b
#owner: https://api.github.com/users/JackWillz

# An aggregation dictionary allows you to do the mean for some values, and a sum for others. 
# In our case we want to the average of all the columns, but count the number of games per champion
agg_dict = {'LaneType': 'count'}
for col in df.columns[4:]:
    agg_dict[col] = 'mean'
    
avg_champ_df = df.groupby(['Lane', 'Champion']).agg(agg_dict).reset_index()