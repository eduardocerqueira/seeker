#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

import logging
from django.urls import path, include
from rest_framework.routers import DefaultRouter, SimpleRouter

# Core viewsets
from lyndsy.users.api.profiles.views import UserProfileViewSet, AddressViewSet
from lyndsy.users.api.invitations.views import InvitationViewSet

# Auth (OAuth‑only)
from lyndsy.users.api.auth.views import ValidateSessionView, LogoutView
from lyndsy.users.api.auth.oauth_views import (
    StoreRefreshTokenView,
    RefreshTokenView,
    RevokeTokenView,
    UserInfoView,
)

logger = logging.getLogger(__name__)

# ─────────────────────────── Routers ────────────────────────────
router = SimpleRouter()
router.register("users", UserProfileViewSet)
router.register("addresses", AddressViewSet, basename="address")
router.register("invitations", InvitationViewSet, basename="invitation")

logger.info("Registered routes: %s", router.urls)

# ─────────────────────── URL patterns ───────────────────────────
urlpatterns = [
    # Router‑generated CRUD endpoints
    path("", include(router.urls)),

    # Extra invitation create shortcut
    path(
        "invitations/",
        InvitationViewSet.as_view({"post": "create"}),
        name="invitation-create",
    ),

    # Profile helpers
    path(
        "profiles/me/",
        UserProfileViewSet.as_view({"get": "me", "patch": "me", "put": "me"}),
        name="profile-me",
    ),
    path(
        "users/me/address/",
        AddressViewSet.as_view({"get": "my_address"}),
        name="my-address",
    ),
    path(
        "users/<uuid:pk>/role/",
        UserProfileViewSet.as_view({"patch": "set_role"}),
        name="user-role",
    ),
    path(
        "users/<uuid:user_id>/address/",
        AddressViewSet.as_view({"get": "retrieve", "patch": "update", "put": "update"}),
        name="user-address",
    ),

    # Session utilities
    path("validate-session/", ValidateSessionView.as_view(), name="validate_session"),
    path("logout/", LogoutView.as_view(), name="logout"),

    # OAuth 2.0 custom endpoints
    path("oauth/store-refresh/", StoreRefreshTokenView.as_view(), name= "**********"
    path("oauth/refresh/", RefreshTokenView.as_view(), name= "**********"
    path("oauth/revoke/", RevokeTokenView.as_view(), name= "**********"
    path("oauth/userinfo/", UserInfoView.as_view(), name="oauth_userinfo"),

    # App‑specific API namespaces
    path("users/", include("lyndsy.users.api.urls")),
    path("healthplans/", include("lyndsy.healthplans.api.urls")),
    path("support/", include("lyndsy.support.urls", namespace="support")),
    path("billing/", include("lyndsy.billing.api.urls", namespace="billing")),
]

app_name = "api"
app_name = "api"
