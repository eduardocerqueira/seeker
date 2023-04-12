#date: 2023-04-12T17:05:13Z
#url: https://api.github.com/gists/b4b213c0c5479a998fa8ef6c856595a1
#owner: https://api.github.com/users/srang992

# listing all the tastes
tastes = list(taste_encode.columns) 
taste_dict = {}

# taking the sum of the values of those taste columns to understand how many people are agreed with that taste
for taste in tastes: 
  taste_dict[taste] = sum(taste_encode[taste]) 
  
# sorting the taste dictionary in decending order
taste_dict = sort_sliced_dict(taste_dict, is_reverse=True, item_count=8)

# custmizing and plotting the data
fig = go.Figure(data=[go.Pie(labels=list(taste_dict.keys()), values=list(taste_dict.values()), pull=[0.1, 0, 0, 0])]) 
fig.update_traces(textinfo='percent+label', textposition='inside') 
fig.update_layout(uniformtext_minsize=12, title={'text': "Most Memorable Taste"}) 
fig.show()