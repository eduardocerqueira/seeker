#date: 2022-07-07T17:14:17Z
#url: https://api.github.com/gists/799f3729a8fd84ccc0d3e11cb9bcf5b0
#owner: https://api.github.com/users/abrahamzetz

cols = spotify_df.loc[:,'danceability':'tempo'].columns.to_list()
colors = ['#a3eee3', '#ea8e0f', '#b7dfbb', '#db6992', '#e4b459', '#a5c8d4']

for col in cols:
    fig, ax = plt.subplots(figsize=(20, 16))
    ax = sns.violinplot(x='source', y=col, data=spotify_df, palette=colors)
    plt.title(f'Song Distribution per Daily Mix Category Based on {col.title()}')
    plt.ylabel(col.title())
    plt.xlabel(None)
    plt.tight_layout()
    plt.show()
    print('{}\nMin: {}\nMax: {}'.format(col.title(), min(spotify_df[col]), max(spotify_df[col])))