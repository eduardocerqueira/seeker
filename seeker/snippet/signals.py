#date: 2023-03-03T17:00:33Z
#url: https://api.github.com/gists/df34877861860e1ccb0e59b6f0f2ff84
#owner: https://api.github.com/users/docsallover

from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from myapp.models import MyModel

@receiver(pre_save, sender=MyModel)
def my_pre_save_handler(sender, **kwargs):
    # Do something before the model is saved
    pass

@receiver(post_save, sender=MyModel)
def my_post_save_handler(sender, **kwargs):
    # Do something after the model is saved
    pass