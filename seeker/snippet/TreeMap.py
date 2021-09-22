#date: 2021-09-22T17:07:09Z
#url: https://api.github.com/gists/f3d66ddd34f2601970eb93ff155caf2b
#owner: https://api.github.com/users/AveryData

# tree map
df = pd.read_excel("Airline Delays.xlsx")
df_airline = df.groupby(['Airline'], as_index=False).agg({'Arrival Delay':'mean', 'Month':'count'}).rename(columns={'Month':'Count'})

fig=px.treemap(data_frame=df_airline, path = [px.Constant('All'),'Airline'], values='Count', color='Arrival Delay', color_continuous_scale=['blue', 'LightGray', 'red'])
fig.update_layout(layout, width=700, title_text='Airline Flight Count Colored by Delay')
py.plot(fig, filename='jmp-treemap')
fig.show()