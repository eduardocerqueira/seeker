#date: 2022-06-14T16:55:50Z
#url: https://api.github.com/gists/c9a9f8783403950fa9e8acb31c515888
#owner: https://api.github.com/users/FernandoPrzGmz

from django.conf.urls.i18n import i18n_patterns
from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('proyecto.apps.app_1.api.urls')),
    # ...  
    path('api/', include('proyecto.apps.app_n.api.urls')),
]

urlpatterns += i18n_patterns(
    path('', include('proyecto.apps.app_1.urls')),
    # ...
    path('', include('proyecto.apps.app_n.urls')),
  
    # XXX: El LocaleMiddleware de Django no permite cambiar si encuentra el idioma en cabecera por esta opcion
    prefix_default_language=False,
)
