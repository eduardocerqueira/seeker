#date: 2022-10-25T17:36:45Z
#url: https://api.github.com/gists/17a52c00c5f2b351502b98795dd01a29
#owner: https://api.github.com/users/code-and-dogs

fig = make_subplots(
    rows=6, cols=1
    )

fig.add_trace(go.Scatter(x=dfLiverpool['createdAt'], y=dfLiverpool['retweets'],
              name="Liverpool FC Retweets"),
              row=1, col=1)

fig.add_trace(go.Scatter(x=dfManU['createdAt'], y=dfManU['retweets'],
              name="Manchester United Retweets"),
              row=2, col=1)

fig.add_trace(go.Scatter(x=dfArsenal['createdAt'], y=dfArsenal['retweets'],
              name="Arsenal Retweets"),
              row=3, col=1)

fig.add_trace(go.Scatter(x=dfManCity['createdAt'], y=dfManCity['retweets'],
              name="Manchester City Retweets"),
              row=4, col=1)

fig.add_trace(go.Scatter(x=dfSpurs['createdAt'], y=dfSpurs['retweets'],
              name="Tottenham Hotspur Retweets"),
              row=5, col=1)

fig.add_trace(go.Scatter(x=dfChelsea['createdAt'], y=dfChelsea['retweets'],
              name="Chelsea Retweets"),
              row=6, col=1)

fig.update_layout(height=500, width=700,
                  title_text="Premier League Twitter Analysis - Retweet Count")

fig.show()