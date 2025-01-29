#date: 2025-01-29T17:12:32Z
#url: https://api.github.com/gists/4dacfc6a8f242c7d07f925bf6fd64eec
#owner: https://api.github.com/users/andrealmeidaa

from django.urls import path
from .views import index,listar_eventos,listar_atividades,exibir_form_inscricao,registrar_inscricao
app_name='appeventos'
urlpatterns=[
    path('',index,name='index'),
    path('eventos/listar',listar_eventos,name='listar_eventos'),
    path('eventos/<int:evento_id>/atividades/listar',listar_atividades,name='listar_atividades'),
    path('eventos/atividades/<int:atividade_id>/inscrever',exibir_form_inscricao,name='exibir_form_inscricao'),
    path('eventos/atividades/<int:atividade_id>/registrar',registrar_inscricao,name="registrar_inscricao"),
]