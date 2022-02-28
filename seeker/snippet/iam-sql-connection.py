#date: 2022-02-28T17:10:16Z
#url: https://api.github.com/gists/19f453b3f3290041b8592d0576ad8fb9
#owner: https://api.github.com/users/mikeoverjet

db_host_name = "PROJECT:LOCATION:INSTANCE"
db_username = "iam@example-user.com"
db_database = "database_name"

connection = connector.connect(
    db_host_name,
    "pg8000",
    user=db_username,
    password='iam_user', # leave as is password will be pulled from local credentials
    db=db_database,
    enable_iam_auth=True
)