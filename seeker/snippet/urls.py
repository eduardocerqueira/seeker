#date: 2023-03-23T17:06:11Z
#url: https://api.github.com/gists/4fd8f3a5b58d2fd71b79c3189f3720ae
#owner: https://api.github.com/users/Lord-sarcastic

from . import views

app_name = "users"

urlpatterns = [
    path("login/", views.LoginAPIView.as_view(), name="login"),
]