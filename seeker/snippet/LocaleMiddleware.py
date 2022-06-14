#date: 2022-06-14T16:55:50Z
#url: https://api.github.com/gists/c9a9f8783403950fa9e8acb31c515888
#owner: https://api.github.com/users/FernandoPrzGmz

from django.middleware.locale import LocaleMiddleware as LocaleMiddleware_
from django.urls import resolve, URLResolver, get_urlconf
from django.utils import translation

from rest_framework.views import APIView


class LocaleMiddleware(LocaleMiddleware_):
    """
    Clase LocalMiddleware custom para permitir la traducción de los endpoints del API.

    Se obtiene el idioma de forma tradicional de 'django.middleware.locale.LocaleMiddleware' con
    la condicion de  que cuando la consulta se realiza a un subclase de APIView se toma el lenguaje
    directamente desde el header 'HTTP_ACCEPT_LANGUAGE'.
    """


    def __is_api_call(self, request):
        """
        Consulta si el llamado se realiza a un endpoint del API REST para obtener el idioma desde la cabecera
        'Accept-Language'.

        Returns
        -------
        bool
            Retorna True cuando se trata de una llamada a un endpoint del API REST
        """
        # Determinamos si se trata de una llamada al endpoint cuando la clase es una subclase de APIView de
        # rest_framework
        resolver_match = resolve(request.path_info)
        try:
            return issubclass(resolver_match.func.cls, APIView)
        except AttributeError:
            return False

    def process_request(self, request):
        super(LocaleMiddleware, self).process_request(request)

        # Cuando se trata de una llamada a un endpoint/recurso del API REST entonces obtenemos el idioma desde los
        #  headers  para establecerlo como el idioma activo.
        # 127.0.0.1/api/usuarios, le pasas el header y si va a funcionar porque es de Django REST Framework
        # 127.0.0.1/lista-usuarios le pasas el header no va a funcionar porque es una vioew de Django
        language = request.META.get('HTTP_ACCEPT_LANGUAGE')
        if language and self.__is_api_call(request):
            translation.activate(language)
            request.LANGUAGE_CODE = translation.get_language()
