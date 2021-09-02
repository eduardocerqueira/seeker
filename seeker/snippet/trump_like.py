#date: 2021-09-02T17:15:30Z
#url: https://api.github.com/gists/08a27f37de66423091f50abb4ee91341
#owner: https://api.github.com/users/susanli2016

df_clean['fttrump_level'] = pd.cut(df_clean['fttrump1'], bins=[-0.1, 40, 70, 100], labels=('dislike', 'neutral', 'like'))
df_clean.loc[df_clean['vote'] == 'Donald Trump']['fttrump_level'].value_counts(normalize=True)