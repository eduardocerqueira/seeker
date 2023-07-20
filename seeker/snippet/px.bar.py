#date: 2023-07-20T16:56:00Z
#url: https://api.github.com/gists/cfd25a71dd2a3c201a81e47268cc73e7
#owner: https://api.github.com/users/ShahaneMirzoian

long_df = px.data.medals_long()
colors=('#F319A7','#3D429F', '#6DD2CD', '#19A7F3','#F5D7E2')

fig = px.bar(long_df, 
             x="nation", 
             y="count", 
             color="medal", 
             color_discrete_sequence=colors, 
             text_auto=True,
             barmode='group',
             height=390, width=550, 
             template='simple_white')
fig.show()