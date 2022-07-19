#date: 2022-07-19T17:20:17Z
#url: https://api.github.com/gists/93e8619a44dc72489804a968e45e620c
#owner: https://api.github.com/users/rooflexx

from django.db import models


class Person(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)


class Book(models.Model):
    name = models.CharField(max_length=100)
    author = models.ForeignKey(Person, on_delete=models.CASCADE)
