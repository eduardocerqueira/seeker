#date: 2021-09-16T16:52:05Z
#url: https://api.github.com/gists/7f63e3ce8148cc46a8a71c86a6b24697
#owner: https://api.github.com/users/FahadulShadhin

from django.db import models

# The model Country
class Country(models.Model):
  name = models.CharField(max_length=50)
  
  def __str__(self):
        return "%s the place" % self.name

# The model Capital 
class Capital(models.Model):
  country = models.OneToOneField(
        Country,
        on_delete=models.CASCADE,
        primary_key=True,
    )
  
  name = models.CharField(max_length=50)
  
  def __str__(self):
        return "%s the place" % self.name