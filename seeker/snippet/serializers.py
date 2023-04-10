#date: 2023-04-10T16:44:31Z
#url: https://api.github.com/gists/966ce46d2e26e381e1cb5b734bb55e18
#owner: https://api.github.com/users/guynikan

from rest_framework import serializers
from leilao.models import Lote
from django.db.models import F


class LoteSerializer(serializers.ModelSerializer):
    valor_atual_lote = serializers.SerializerMethodField()
    categoria = serializers.CharField(read_only=True)


    class Meta:
        model = Lote
        fields = ['valor_atual_lote', 'classificacao', 'capa', 'titulo', 'encerrado', 'retirado', 'get_absolute_url', 'categoria']

    def valor_atual_lote(self, obj):
        return obj.valor_atual_lote()

    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset.annotate(categoria=F('veiculos__modelo__nome')).order_by('categoria', 'classificacao')