#date: 2022-07-19T17:23:32Z
#url: https://api.github.com/gists/87bd026e76ec25c32301cff8e4479f4c
#owner: https://api.github.com/users/rooflexx

from django.contrib import admin
from example_app.models import *


class PersonAdmin(admin.ModelAdmin):
    pass


admin.site.register(Person, PersonAdmin)
