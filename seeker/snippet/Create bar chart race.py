#date: 2022-05-19T17:35:38Z
#url: https://api.github.com/gists/9beea07b0951fd0f776547e2d4fe9d21
#owner: https://api.github.com/users/insightsbees

#Create bar race animation
my_raceplot = barplot(df,  item_column='Name', value_column='Revenue', time_column='Year',top_entries=20)
my_raceplot.plot(item_label = 'Top 20 Companies', value_label = 'Revenue', frame_duration = 200, date_format='%Y',orientation='horizontal')