#date: 2023-07-20T16:57:54Z
#url: https://api.github.com/gists/29267e502329823a6eaa584f91f53023
#owner: https://api.github.com/users/ShahaneMirzoian

df = px.data.tips()
colors=('#F319A7','#6DD2CD')

fig = px.histogram(df, 
                   x="total_bill", 
                   color="sex", 
                   barmode='overlay',
                   barnorm='percent', 
                   opacity=0.4, 
                   color_discrete_sequence=colors, 
                   height=390, width=550, 
                   template='simple_white')
fig.show()