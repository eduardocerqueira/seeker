#date: 2024-06-25T16:39:19Z
#url: https://api.github.com/gists/79967f9afabd312449bbeb9918f69939
#owner: https://api.github.com/users/L-narendar-kumar

# models.py (in your Django app)
from django.db import models

class Movie(models.Model):
    title = models.CharField(max_length=255)
    # ... other fields you need
    poster_url = models.URLField()  # Field for the poster URL