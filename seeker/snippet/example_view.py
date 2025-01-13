#date: 2025-01-13T16:41:17Z
#url: https://api.github.com/gists/ade4441d41d5f6f32041fac223dc0413
#owner: https://api.github.com/users/TGoddessana

# simple usage..

# only Superuser can access this view..
class CityCreateView(SuperuserRequiredMixin, CreateView):
    model = City
    form_class = CityCreateForm
    template_name = "analyzer/city-form.html"
    success_url = reverse_lazy("city-list")
