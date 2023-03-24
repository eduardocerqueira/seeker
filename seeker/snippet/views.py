#date: 2023-03-24T17:00:58Z
#url: https://api.github.com/gists/a717a2798b2246623c4934eb513e669d
#owner: https://api.github.com/users/chrisgrande

from mixins import RedirectOn404Mixin
from django.views.generic import DetailView


class TestDetail(RedirectOn404Mixin, DetailView):
    model = Test
    redirect_url_name = 'test-index'