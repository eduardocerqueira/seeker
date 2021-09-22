#date: 2021-09-22T17:04:32Z
#url: https://api.github.com/gists/4d00b8b8651b8ae5c964c59a76c9fa0e
#owner: https://api.github.com/users/AveryData

# heatmap
df = pd.read_excel("Airline Delays.xlsx")
df_mean = df.groupby(['Month', 'Day of Month'], as_index=False).agg({'Arrival Delay':'mean'})
df_mean['Month'] = pd.Categorical(df_mean['Month'], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
df_mean = df_mean.sort_values(by=['Month', 'Day of Month'])
df_heatmap = df_mean.pivot(index='Month', columns='Day of Month', values='Arrival Delay')
fig = px.imshow(df_heatmap, x=df_heatmap.columns, y=df_heatmap.index, color_continuous_scale=['blue', 'LightGray', 'red'])
fig.update_layout(layout, title_text='Arrival Delays by Day')
fig.update_yaxes(autorange="reversed")
py.plot(fig, filename='jmp-heatmap')
fig.show()