#date: 2022-11-10T17:10:40Z
#url: https://api.github.com/gists/fb5674769d59dcf7380045a99fa74488
#owner: https://api.github.com/users/paudelgaurav

from django.db import models


class Product(models.Model):
    title = models.CharField(max_length=50)
    description = models.TextField(blank=True)
    note = models.TextField(blank=True)

    def __str__(self) -> str:
        return self.title
