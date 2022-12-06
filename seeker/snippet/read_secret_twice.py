#date: 2022-12-06T17:00:11Z
#url: https://api.github.com/gists/380777c8a3e3d8e17106e360324c9eb0
#owner: https://api.github.com/users/ShahriyarR

>>> obj.get_secret()
'awesome_password'
>>> obj.get_secret()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/shako/REPOS/py-read-once/.venv/lib/python3.10/site-packages/readonce.py", line 47, in get_secret
    raise UnsupportedOperationException("Sensitive data was already consumed")
readonce.UnsupportedOperationException: ('Not allowed on sensitive value', 'Sensitive data was already consumed')