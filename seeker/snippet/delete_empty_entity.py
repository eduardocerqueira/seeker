#date: 2022-09-22T17:20:42Z
#url: https://api.github.com/gists/85f1d83778cd50fabee057c2fc41ff31
#owner: https://api.github.com/users/raghavanandan7

from eshares.corporations.models import Corporation


CORP_ID = 2267409

Corporation.objects.get(id=CORP_ID).delete()