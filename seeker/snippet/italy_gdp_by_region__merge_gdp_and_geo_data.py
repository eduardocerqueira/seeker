#date: 2025-04-25T16:38:08Z
#url: https://api.github.com/gists/0cae1d7a4891a3b5b4ecfac8de82437b
#owner: https://api.github.com/users/bianconif

year = 2023
df_merged = pd.merge(left=df_gdata, right=df_gdp[[str(year), 'Region']], left_on='reg_name', right_on='Region')

#Convert the values in EUR
df_merged[str(year)] = 1000*df_merged[str(year)]

print(df_merged.head())