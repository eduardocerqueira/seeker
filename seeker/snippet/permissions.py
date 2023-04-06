#date: 2023-04-06T17:07:52Z
#url: https://api.github.com/gists/a147cb965280269c63899e2ece9ed648
#owner: https://api.github.com/users/GKORus

from rest_framework import permissions


class IsAuthorOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.author == request.user