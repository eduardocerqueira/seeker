#date: 2023-07-20T16:57:02Z
#url: https://api.github.com/gists/0507a1cccab6b9479f08a4baf24e2318
#owner: https://api.github.com/users/ShahaneMirzoian

df = px.data.gapminder()\
    .groupby(by=['year', 'continent'], 
             as_index=False).agg({'pop': sum})\
    .sort_values(by=['year', 'pop'], ascending=[True, False])

colors=('#F319A7','#3D429F', '#6DD2CD', '#19A7F3','#F5D7E2')

fig = px.area(df, x='year', 
              y='pop',
              color='continent',
              color_discrete_sequence=colors
              height=390, width=550, 
              template='simple_white', 
              line_shape='spline')
fig.show()