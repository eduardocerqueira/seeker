#date: 2023-07-20T16:58:47Z
#url: https://api.github.com/gists/adf0fc7ded92c931338652b3a0e93a88
#owner: https://api.github.com/users/ShahaneMirzoian

df = px.data.gapminder()\
    .groupby(by=['year', 'continent'], 
             as_index=False).agg({'pop': sum})\
    .sort_values(by=['year', 'pop'], ascending=[True, False])

colors=('#F319A7','#3D429F', '#6DD2CD', '#19A7F3','#F5D7E2')

fig = px.histogram(df, x='year', 
              y='pop',
              color='continent',
              color_discrete_sequence=colors,
              barnorm='percent',
              nbins=12,
              height=390, width=550, 
              template='simple_white')

fig.show()