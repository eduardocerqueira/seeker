#date: 2023-01-30T16:38:21Z
#url: https://api.github.com/gists/98b9bb84beff495ae54642b46402565f
#owner: https://api.github.com/users/bukowa

from functools import wraps

import django.http
import django.views.static


def no_cache_static(f):
    @wraps(f)
    def static(*a, **kw):
        response: django.http.response.HttpResponse = f(*a, **kw)  # type:
        response.headers["Cache-Control"] = "no-cache"
        return response

    return static


django.views.static.serve = no_cache_static(django.views.static.serve)
