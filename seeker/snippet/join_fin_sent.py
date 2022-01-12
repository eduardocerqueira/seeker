#date: 2022-01-12T17:15:05Z
#url: https://api.github.com/gists/170c2ba2b03afae2234e26e1f5596051
#owner: https://api.github.com/users/haykaza

#set index columns to implement dataframe joining
all_data.set_index('Instrument', inplace = True)
sentiments.rename(columns = {"RIC": "Instrument"}, inplace=True)
sentiments.set_index('Instrument', inplace = True)

#join NLP based and financial variable datasets
all_data = all_data.join(sentiments, on = 'Instrument', how = 'inner').reset_index().drop_duplicates(subset = ['Instrument'], keep = 'first')