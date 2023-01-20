#date: 2023-01-20T17:10:49Z
#url: https://api.github.com/gists/5b1f47b09a03ff455e8d113126a0f182
#owner: https://api.github.com/users/jgoodie

host = '192.168.xxx.yyy'
user = 'dbc'
pw = 'dbc'

conn = "**********"=host, user=user, password=pw)
cur = conn.cursor()
cur.execute("{fn teradata_nativesql}{fn teradata_autocommit_on}")
conn.close()lose()