#date: 2022-05-02T16:48:15Z
#url: https://api.github.com/gists/c5b9eaf1c3da1619a6629a2ac7277c86
#owner: https://api.github.com/users/m33ch33

>>> sha256(json.dumps('{"a":"a"}').encode("utf-8")).hexdigest()
'c4602ab462c46e37f6677e339371bd86337051838ce1c49d9ff4b91ffffa67c4'

sha256(json.dumps('{"a":  "a"}').encode("utf-8")).hexdigest()
'e37d578267a7f6d8936b8a749f1eaedcde9434a01ddb6333c2d60646d87c8132'

sha256('{"a":"a"}'.encode("utf-8")).hexdigest()
'681523631e0f5d3904d881dd163683081e0e45afdad34376ff5bf5fbadada6c7'

>>> sha256(str(json.loads('{"a":"a"}')).encode("utf-8")).hexdigest()
'8e9ee3e5afddd3561dc6e1d693ad3c90a711c579621d755447e07c357a311450'
