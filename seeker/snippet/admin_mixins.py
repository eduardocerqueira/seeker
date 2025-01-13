#date: 2025-01-13T16:41:17Z
#url: https://api.github.com/gists/ade4441d41d5f6f32041fac223dc0413
#owner: https://api.github.com/users/TGoddessana

from django.contrib.auth.mixins import AccessMixin


class SuperuserRequiredMixin(AccessMixin):
    """
    CBV mixin which verifies that the current user is a superuser.
    """

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_superuser:
            return self.handle_no_permission()
        return super().dispatch(request, *args, **kwargs)


class StaffRequiredMixin(AccessMixin):
    """
    CBV mixin which verifies that the current user is staff.
    """

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return self.handle_no_permission()
        return super().dispatch(request, *args, **kwargs)
