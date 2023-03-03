#date: 2023-03-03T16:50:27Z
#url: https://api.github.com/gists/2a33c5d033e6e72c521804bdb2db96d7
#owner: https://api.github.com/users/docsallover

from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=200)
    published_date = models.DateField()
    num_pages = models.IntegerField()