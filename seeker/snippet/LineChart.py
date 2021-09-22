#date: 2021-09-22T17:15:22Z
#url: https://api.github.com/gists/aac312af61292bc3998da1e38b2f4970
#owner: https://api.github.com/users/AveryData

# Line Chart 
jmp_colors = ["blue", "red", "green", "BlueViolet"]
df = pd.read_excel("Australian Tourism.xlsx")

fig = px.line(data_frame=df, x='Year', y="Room occupancy rate (%)", color='Quarter', template='simple_white', width=700, color_discrete_sequence=jmp_colors)
fig.update_layout(layout, title_text = 'Room occupancy rate vs. Year and Quarter')
fig.update_xaxes(mirror=True)
fig.update_yaxes(mirror=True)
py.plot(fig, filename='jmp-basic graph builder')
fig.show()