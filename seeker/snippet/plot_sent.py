#date: 2022-01-12T17:11:52Z
#url: https://api.github.com/gists/7fcf353381d06e58c3b096a06afbfb3a
#owner: https://api.github.com/users/haykaza

#plot values using Plotly scatter plot
fig = px.scatter(sentiments, y = ["SentOverallFBert", "SentOverallLabs"])
#update plot layout
fig.update_layout(height=700, width=1100)
fig.update_yaxes(range=[-150, 550], tick0=0)

#move legend to the top left
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))
fig.show()