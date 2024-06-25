#date: 2024-06-25T16:39:19Z
#url: https://api.github.com/gists/79967f9afabd312449bbeb9918f69939
#owner: https://api.github.com/users/L-narendar-kumar

from django.urls import path
from . import views

urlpatterns = [
    path('', views.movie_recommendations, name='movie_recommendations'),
    # ... any other URL patterns you might have for your app
]