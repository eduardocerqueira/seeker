#date: 2023-10-13T17:07:14Z
#url: https://api.github.com/gists/b47fdf2874d73696512b1f07e5940beb
#owner: https://api.github.com/users/SudhenduP

fig = px.line_3d(
        filtered_df1,
        x="X_OFFSET",
        y="Y_OFFSET",
        z="STATION_TVD",
        template="plotly_dark",
        hover_name="UWI",
    )
    fig.update_layout(width=800, 
              height=800,
              autosize=False,
              scene={'zaxis':{'autorange':'reversed'}})

fig.update_traces(line={'width':10})
st.plotly_chart(fig,use_container_width=False)