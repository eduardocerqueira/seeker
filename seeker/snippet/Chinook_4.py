#date: 2022-02-25T16:59:09Z
#url: https://api.github.com/gists/11389b002316cf3ae75ca02c70d867bf
#owner: https://api.github.com/users/ssime-git

conn = sqlite3.connect('chinook.db')
conn.row_factory = dict_factory
cur = conn.cursor()

results = cur.execute(query, to_filter).fetchall()