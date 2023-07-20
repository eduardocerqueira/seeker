#date: 2023-07-20T16:53:50Z
#url: https://api.github.com/gists/5f531469c0bb3df244d5a1755ca9dfba
#owner: https://api.github.com/users/ShahaneMirzoian

df = px.data.stocks() 
colors=('#F319A7','#3D429F', '#6DD2CD', '#19A7F3','#F5D7E2')

fig = px.line(df[df.date>'2019-01-01'], 
              x='date', 
              y=["GOOG", "AAPL", "AMZN"], 
              color_discrete_sequence=colors, 
              height=390, width=550, 
              template='simple_white')
fig.show()