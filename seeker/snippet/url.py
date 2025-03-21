#date: 2025-03-21T17:08:00Z
#url: https://api.github.com/gists/40ce302c33c3286b1f461bfd69458a1a
#owner: https://api.github.com/users/jterjjj65

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'categories', views.CategoryViewSet)
router.register(r'products', views.ProductViewSet)

app_name = 'catalog'

urlpatterns = [
    path('', include(router.urls)),
    path('attributeoption/get_options/', views.get_attribute_options, name='get_attribute_options'),
    path('create/', views.create_product, name='create_product'),
]