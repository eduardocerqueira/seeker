#date: 2023-03-24T17:00:58Z
#url: https://api.github.com/gists/a717a2798b2246623c4934eb513e669d
#owner: https://api.github.com/users/chrisgrande

from django.http import Http404
from django.urls import reverse
from django.shortcuts import redirect
from django.views.generic.detail import SingleObjectMixin


class RedirectOn404Mixin(SingleObjectMixin):
    redirect_url_name = None

    def get(self, request, *args, **kwargs):
        try:
            return super().get(request, *args, **kwargs)
        except Http404:
            if self.redirect_url_name:
                redirect_url = reverse(self.redirect_url_name)
                return redirect(redirect_url)
            else:
                raise