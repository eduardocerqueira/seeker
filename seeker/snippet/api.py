#date: 2022-11-10T17:18:03Z
#url: https://api.github.com/gists/00e137fbc955101aec1f8fec378ddd54
#owner: https://api.github.com/users/paudelgaurav

from rest_framework.viewsets import ReadOnlyModelViewSet
from rest_framework.serializers import ModelSerializer

from .models import Product


class ProductSerializer(ModelSerializer):
    class Meta:
        model = Product
        fields = "__all__"


class ProductReadOnlyViewSet(ReadOnlyModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
