#date: 2024-03-04T17:01:56Z
#url: https://api.github.com/gists/5899f6548e57c4e1181d54034c26b7d8
#owner: https://api.github.com/users/aevosolar

from celery import shared_task
from .models import Age
from random import randint
from django.db import transaction

@shared_task
def ageRandom():
    with transaction.atomic():
        age = randint(0,101)
        age_model = Age(age=age)
        age_model.save()
        print(f"Salvado ? age:{age}")