#date: 2022-02-16T16:54:26Z
#url: https://api.github.com/gists/f200f4fd62b63ffb205b0f9d55f0d8b5
#owner: https://api.github.com/users/fabricius1

from django.db import models


class Genre(models.Model):
    name = models.CharField(max_length=150)
    
    def __str__(self):
        return self.name


class Film(models.Model):
    title = models.CharField(max_length=200)
    year = models.PositiveIntegerField()
    genre = models.ForeignKey(Genre, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.title
