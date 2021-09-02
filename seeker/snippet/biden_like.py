#date: 2021-09-02T17:14:10Z
#url: https://api.github.com/gists/cb76bf8417ce89e9478f5b142c57751a
#owner: https://api.github.com/users/susanli2016

df_clean['ftbiden_level'] = pd.cut(df_clean['ftbiden1'], bins=[-0.1, 40, 70, 100], labels=('dislike', 'neutral', 'like'))
df_clean.loc[df_clean['vote'] == 'Joe Biden']['ftbiden_level'].value_counts(normalize=True)