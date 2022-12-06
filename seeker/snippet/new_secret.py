#date: 2022-12-06T17:08:00Z
#url: https://api.github.com/gists/f53fb54c77e454653e9dc92238faa014
#owner: https://api.github.com/users/ShahriyarR

>>> obj = "**********"="awesome_password")
>>> obj.add_secret("new_fake_date")
>>> obj.get_secret()
'new_fake_date'
>>> obj.get_secret()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/shako/REPOS/py-read-once/.venv/lib/python3.10/site-packages/readonce.py", line 47, in get_secret
    raise UnsupportedOperationException("Sensitive data was already consumed")
readonce.UnsupportedOperationException: ('Not allowed on sensitive value', 'Sensitive data was already consumed')med')