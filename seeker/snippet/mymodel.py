#date: 2024-05-08T17:06:49Z
#url: https://api.github.com/gists/96d19b0e7bbd6ff4ed7fa5462ed41ef1
#owner: https://api.github.com/users/meyt

from django.db import models
from rest_framework import serializers

class MyModel(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
      

class MyModelSerializer(NestedModelSerializer):
    class Meta:
        model = MyModel
        fields = '__all__'
