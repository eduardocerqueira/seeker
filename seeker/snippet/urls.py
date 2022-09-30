#date: 2022-09-30T17:14:38Z
#url: https://api.github.com/gists/4e182a4d40266886bd8ee81c8cb9cb05
#owner: https://api.github.com/users/gadiazsaavedra

"""PrimerMVT URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from AppPrimerMVT.views import mi_familia
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("AppPrimerMVT/", mi_familia),
    # path("AppPrimerMVT/", include("AppPrimerMVT.urls")),
]
