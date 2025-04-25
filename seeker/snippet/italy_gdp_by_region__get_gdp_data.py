#date: 2025-04-25T16:33:27Z
#url: https://api.github.com/gists/794c88147540996372b13958724ae708
#owner: https://api.github.com/users/bianconif

df_gdp = pd.read_csv('https://raw.githubusercontent.com/bianconif/graphic_communication_notebooks/refs/heads/master/data/istat/italy-gdp-by-region.csv', comment='#')
print(df_gdp.head()) 