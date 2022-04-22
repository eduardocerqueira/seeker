#date: 2022-04-22T17:20:40Z
#url: https://api.github.com/gists/677ed1203e588127673cde2165e2c0a4
#owner: https://api.github.com/users/odhiambo123


# Register your models here.
from django.contrib import admin
from .models import Post, Comments, Like

admin.site.register(Post)
admin.site.register(Comments)
admin.site.register(Like)
