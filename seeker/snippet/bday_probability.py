#date: 2021-11-12T17:17:27Z
#url: https://api.github.com/gists/e6c9ecd7ff1e3a510ee4a0fa50ed1873
#owner: https://api.github.com/users/wmblack23

ppl_in_room, probability = [], []

for i in range(1, 75):
    probability.append(round(1 - (364/365)**((i*(i-1))/2), 4))
    ppl_in_room.append(i)
    
fig = px.line(x = ppl_in_room, y = probability, title='Probabilites of Birthday Paradox', template='plotly_dark')