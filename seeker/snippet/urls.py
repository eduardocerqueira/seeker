#date: 2023-03-03T16:44:52Z
#url: https://api.github.com/gists/590e695aed2ac0aee079a19a42adaab2
#owner: https://api.github.com/users/docsallover

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
]