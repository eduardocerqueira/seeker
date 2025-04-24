#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

"""
Root URLConf for the Lyndsy project.

Key points
-----------
* Explicit sub‑paths for each major API module come **before** the generic
  router so they aren’t shadowed.
* `/oauth2/token/` issues the refresh token in a Secure, HttpOnly cookie
  via `SpaTokenView`; the default DOT endpoints remain under `/o/`.
* Health‑check, robots, sitemap, and debug helpers kept unchanged.
/Users/chrisrodgers/Projects/lyndsy-workspace/lyndsy/config/urls.py
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView
from django.contrib.sitemaps.views import sitemap
from django.http import HttpResponse, JsonResponse
from django.views import defaults as default_views
from django.contrib.auth import views as auth_views
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from lyndsy.blue_button_auth.views import BlueButtonCallback
from lyndsy.core.oauth2.views import SpaTokenView  # custom SPA token endpoint
from lyndsy.core.views import csrf_cookie
from lyndsy.users.views import CustomAdminLoginView
from lyndsy.sitemaps import StaticViewSitemap

# Callback URL group
from config.urls_callbacks import urlpatterns as callback_urlpatterns


# ─────────────────────────────  Utility views  ──────────────────────────────
def healthcheck(request):
    return HttpResponse("OK")


def api_root(request):
    return JsonResponse(
        {
            "status": "ok",
            "message": "API is operational",
            "frontend_url": "https://informedpluschoice.com/",
            "documentation_url": request.build_absolute_uri("/api/docs/"),
        }
    )


def robots_txt(request):
    lines = [
        "User-agent: *",
        "Disallow: /admin/",
        "Disallow: /accounts/",
        "Allow: /",
        "",
        f"Sitemap: {request.build_absolute_uri('/sitemap.xml')}",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")


sitemaps = {"static": StaticViewSitemap}

# ───────────────────────────  URL patterns  ────────────────────────────────
urlpatterns = [
    # ── Core / utility ─────────────────────────────────────────────────────────
    path(settings.ADMIN_URL, admin.site.urls),
    path("", api_root, name="home"),
    path("healthcheck/", healthcheck, name="healthcheck"),
    path("favicon.ico", RedirectView.as_view(
        url=settings.STATIC_URL + "images/favicons/favicon.png"), name="favicon"
    ),
    path("robots.txt", robots_txt, name="robots_txt"),
    path("sitemap.xml", sitemap, {"sitemaps": sitemaps},
         name="django.contrib.sitemaps.views.sitemap"),

    # ── Authentication & accounts ──────────────────────────────────────────────
    path("users/", include("lyndsy.users.urls", namespace="users")),
    path("accounts/", include("django.contrib.auth.urls")),
    path(
        "password_change/",
        auth_views.PasswordChangeView.as_view(
            template_name= "**********"
            success_url="/",
        ),
        name= "**********"
    ),

    # ── OAuth2 endpoints ───────────────────────────────────────────────────────
    path("oauth2/token/", SpaTokenView.as_view(), name= "**********"
    path("o/", include("oauth2_provider.urls",
                       namespace="oauth2_provider")),                # default DOT

    # ── BlueButton + health‑plan callbacks ─────────────────────────────────────
    path("blue-button/callback/", BlueButtonCallback.as_view(),
         name="blue_button_callback"),
    *callback_urlpatterns,

    # ── API surface ────────────────────────────────────────────────────────────
    path("api/patients/", include("lyndsy.patients.urls")),
    path("api/dashboard/", include("lyndsy.dashboard.urls")),
    path("api/prescriptions/", include("lyndsy.prescriptions.urls")),
    path("api/support/", include("lyndsy.support.urls", namespace="support")),
    path("api/blue-button-auth/", include(
        "lyndsy.blue_button_auth.urls", namespace="blue_button_auth")),
    path("api/contacts/", include("lyndsy.contacts.api.urls")),
    path("api/", include("config.api_router")),      # keep last to avoid shadowing

    # CSRF helper
    path("api/csrf/", csrf_cookie, name="csrf-cookie"),

    # Local dev redirect to SPA
    path("app/login/",
         RedirectView.as_view(url="http://localhost:3000/app/login",
                              permanent=False)),

    # ── API documentation ──────────────────────────────────────────────────────
    path("api/schema/", SpectacularAPIView.as_view(), name="api-schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="api-schema"),
         name="api-docs"),
]

# ── Non‑production extras ─────────────────────────────────────────────────────
if settings.DEBUG or getattr(settings, "TESTING", False):
    urlpatterns += [
        path("api/test/", include("lyndsy.users.tests.api.urls",
                                  namespace="test_api")),
    ]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += [
        path("400/", default_views.bad_request,
             kwargs={"exception": Exception("Bad Request!")}),
        path("403/", default_views.permission_denied,
             kwargs={"exception": Exception("Permission Denied!")}),
        path("404/", default_views.page_not_found,
             kwargs={"exception": Exception("Page not Found!")}),
        path("500/", default_views.server_error),
    ]
    if "debug_toolbar" in settings.INSTALLED_APPS:
        import debug_toolbar
        urlpatterns = [path("__debug__/", include(debug_toolbar.urls))] + urlpatterns

# Custom admin login view
admin.site.login = CustomAdminLoginView.as_view()

admin.site.login = CustomAdminLoginView.as_view()
