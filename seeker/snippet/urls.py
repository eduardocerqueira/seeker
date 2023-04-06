#date: 2023-04-06T17:07:52Z
#url: https://api.github.com/gists/a147cb965280269c63899e2ece9ed648
#owner: https://api.github.com/users/GKORus

from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

from .views import CommentViewSet, GroupViewSet, PostViewSet


router = DefaultRouter()
router.register(r'posts', PostViewSet, basename='posts')
router.register(r'groups', GroupViewSet, basename='groups')
router.register(r'posts/(?P<post_id>[^/.]+)/comments', CommentViewSet, basename='comments')

urlpatterns = [
    path('api-token-auth/', obtain_auth_token),
    path('', include(router.urls)),
]