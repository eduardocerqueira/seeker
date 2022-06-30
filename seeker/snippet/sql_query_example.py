#date: 2022-06-30T17:19:35Z
#url: https://api.github.com/gists/5e340c0465801e27f8858a658775c260
#owner: https://api.github.com/users/hacceebhassan

username = request.get_json(force=True)['username']

sql = (
    "SELECT * FROM "
    + get("DB_NAME") + " WHERE username ='" + username + "'"
)

curr.execute (sql)