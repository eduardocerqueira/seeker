#date: 2021-12-01T17:07:00Z
#url: https://api.github.com/gists/ec3faad5bf551f2fad4a546078d14535
#owner: https://api.github.com/users/JackWillz

avg_champ_df = avg_champ_df[avg_champ_df['Count']>=5000]
top_champ_df = avg_champ_df[avg_champ_df['Lane']=='top']