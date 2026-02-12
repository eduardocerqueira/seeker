#date: 2026-02-12T17:46:36Z
#url: https://api.github.com/gists/0f8f1a7d3f2d4b62584d5c5ab8d10347
#owner: https://api.github.com/users/dryan

from django import http
from django.db.models import F

from app.models import Redirect


class RedirectsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: http.HttpRequest) -> http.HttpResponse:
        response = self.get_response(request)
        if response.status_code == 404:
            redirect = Redirect.objects.filter(path=request.path).first()
            if redirect:
                Redirect.objects.filter(pk=redirect.pk).update(used=F("used") + 1)
                return http.HttpResponseRedirect(redirect.destination)
        return response
