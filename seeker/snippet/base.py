#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

"""
Django configuration for a multi-tenant stack.

* base.py  – shared across every environment (local, staging, prod)
* local.py – overrides for local Docker development

Changes introduced:
• switched DB engine to `django_tenants.postgresql_backend`
• declared SHARED_APPS / TENANT_APPS structure
• added DATABASE_ROUTERS, TENANT_MODEL, TENANT_DOMAIN_MODEL
• added wildcard hosts for *.localtest.me (dev) and *.informedpluschoice.com (prod)
• kept OAuth2 + PKCE workflow unchanged
• ready for Postgres 17 with pg_tde
"""

# ─────────────────────────────  config/settings/base.py  ─────────────────────────────
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import timedelta
from cryptography.fernet import Fernet

import dj_database_url
from django.core.exceptions import ImproperlyConfigured

# helper -------------------------------------------------------------------------

def env(name: str, default: str | None = None, *, required: bool = False):
    value = os.getenv(name, default)
    if required and value is None:
        raise ImproperlyConfigured(f"Env var '{name}' is required")
    return value

# paths --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
APPS_DIR = BASE_DIR / "lyndsy"
if str(APPS_DIR) not in sys.path:
    sys.path.append(str(APPS_DIR))

LOG_DIR = Path(env("DJANGO_LOG_DIR", str(BASE_DIR / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# runtime flags ------------------------------------------------------------------
ENVIRONMENT = env("DJANGO_ENVIRONMENT", "local").lower()
DEBUG = env("DJANGO_DEBUG", "true" if ENVIRONMENT == "local" else "false").lower() in {"1", "true", "yes"}

# secret key ---------------------------------------------------------------------
 "**********"i "**********"f "**********"  "**********"E "**********"N "**********"V "**********"I "**********"R "**********"O "**********"N "**********"M "**********"E "**********"N "**********"T "**********"  "**********"= "**********"= "**********"  "**********"" "**********"l "**********"o "**********"c "**********"a "**********"l "**********"" "**********"  "**********"a "**********"n "**********"d "**********"  "**********"n "**********"o "**********"t "**********"  "**********"e "**********"n "**********"v "**********"( "**********"" "**********"D "**********"J "**********"A "**********"N "**********"G "**********"O "**********"_ "**********"S "**********"E "**********"C "**********"R "**********"E "**********"T "**********"_ "**********"K "**********"E "**********"Y "**********"" "**********") "**********": "**********"
    SECRET_KEY = "**********"
else:
    SECRET_KEY = "**********"=True)

# core settings ------------------------------------------------------------------
TIME_ZONE = env("TIME_ZONE", "America/Los_Angeles")
LANGUAGE_CODE = "en-us"
SITE_ID = int(env("DJANGO_SITE_ID", "1"))
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_I18N = True
USE_TZ = True
LOCALE_PATHS = [str(BASE_DIR / "locale")]

ALLOWED_HOSTS: list[str] = [
    "localhost",
    "127.0.0.1",
    "admin.localhost",
    "::1",
    "*.localtest.me",
]
# --------------------------------------------------------------------------
# CSRF / CORS ‑‑ allow local HTTPS origins during development
# --------------------------------------------------------------------------
CSRF_TRUSTED_ORIGINS = [
    "https://localhost",
    "https://127.0.0.1",
    "https://*.localhost.me",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
# --------------------------------------------------------------------------
# Static files
# Ensure STATIC_URL is always defined before it is referenced in config/urls
STATIC_URL = env("STATIC_URL", "/static/")
# Absolute path where `collectstatic` will gather compiled assets
STATIC_ROOT = BASE_DIR / "staticfiles"

# --------------------------------------------------------------------------
# Optional custom path for Django admin – overridable via env var
ADMIN_URL = env("DJANGO_ADMIN_URL", "admin/")

# Multi-tenancy configuration  ---------------------------------------------------
# SHARED_APPS = apps that are shared across all tenants (in public schema)
# TENANT_APPS = apps that are tenant-specific (in tenant schemas)
SHARED_APPS = [
    # django_tenants and core app will be added in INSTALLED_APPS only
    "lyndsy.organizations",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",  # Uncommented to serve static files from all hosts
    "lyndsy.billing",            # billing is global
    "corsheaders",               # CORS headers are global
    "rest_framework",            # DRF utilities in public schema

    # do NOT add 'users' or 'django.contrib.admin' here
]

TENANT_APPS = [
    # Django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.sites",
    "django.contrib.sitemaps",
    # Staticfiles moved to SHARED_APPS
    # Third-party tenant apps
    "oauth2_provider",
    "rest_framework.authtoken",
    "drf_spectacular",
    "storages",
    "django_celery_beat",
    "django_celery_results",
    "django_filters",
    # Your domain apps
    "lyndsy.users",
    # "rest_framework",  # Removed - already in SHARED_APPS
    "lyndsy.blue_button_auth",
    "lyndsy.providers",
    "lyndsy.healthplans",
    "lyndsy.benefits",
    "lyndsy.support",
    "lyndsy.cask",
    "lyndsy.patients",
    "lyndsy.claims",
    "lyndsy.branding",
    "lyndsy.dashboard",
    "lyndsy.prescriptions",
    "lyndsy.partial_claims",
    "lyndsy.consolidated",
    "lyndsy.contacts",
    "lyndsy.notifications",
]

# Define the installed apps for Django - these two must be first
INSTALLED_APPS = [
    "django_tenants",  # Must be first
    "lyndsy.core",     # Your tenant-aware models
] + SHARED_APPS + TENANT_APPS

# Simple concatenation without the earlier entries
INSTALLED_APPS = ["django_tenants", "lyndsy.core"] + SHARED_APPS + TENANT_APPS


# ─── Multi‑tenancy ───────────────────────────────
TENANT_MODEL = "core.OrganizationTenant"
TENANT_DOMAIN_MODEL = "core.Domain"
DATABASE_ROUTERS = ("django_tenants.routers.TenantSyncRouter",)

TENANT_TYPES = {
    "aco": {
        "APPS": TENANT_APPS,              # the default list
        "URLCONF": "config.urls.tenant",
    },
    "agency": {
        "APPS": TENANT_APPS + ["agent_portal"],
        "URLCONF": "config.urls.tenant",
    },
}

# Authentication configuration
AUTH_USER_MODEL = "users.User"

# middleware ---------------------------------------------------------------------
MIDDLEWARE = [
    "django_tenants.middleware.main.TenantMainMiddleware",  # Must be first
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "config.middleware.cache_control_middleware.CacheControlMiddleware",
    "lyndsy.users.middleware.SingleSessionMiddleware",
    "lyndsy.core.middleware.audit.AuditLogMiddleware",
    "lyndsy.core.oauth2.middleware.RefreshTokenCookieMiddleware",
]

# database -----------------------------------------------------------------------
DATABASES = {
    "default": dj_database_url.parse(
        env("DATABASE_URL", "postgres://postgres:postgres@postgres:5432/lyndsy"),
        conn_max_age=int(env("CONN_MAX_AGE", "60")),
    )
}
DATABASES["default"].update(
    {
        "ENGINE": "django_tenants.postgresql_backend",
        "ATOMIC_REQUESTS": True,
    }
)

# redis cache --------------------------------------------------------------------
REDIS_URL = env("REDIS_URL", "redis://redis:6379/0")
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            # Set to allkeys-lru for fair eviction across tenants
            "EVICTION_POLICY": "allkeys-lru",
        },
    }
}

# logging configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "format": "[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": "INFO",
        },
        "hipaa": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "hipaa_audit.log"),
            "formatter": "console",
            "level": "INFO",
        },
    },
    "loggers": {
        "hipaa_audit": {
            "handlers": ["hipaa"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# rest framework + oauth2 (modified to use custom generator) -------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "oauth2_provider.contrib.rest_framework.OAuth2Authentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    # Per-tenant throttling
    "DEFAULT_THROTTLE_CLASSES": [
        "lyndsy.core.throttling.TenantAwareUserRateThrottle",
        "lyndsy.core.throttling.TenantAwareAnonRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "user": "1000/day",
        "anon": "100/day",
    },
}
# --------------------------------------------------------------------------
# Django template engine (required for admin & debug‑toolbar)
# --------------------------------------------------------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Project‑level templates live in   <repo_root>/templates/
        "DIRS": [BASE_DIR / "templates"],
        # Auto‑discover app‑level templates like  users/templates/users/…
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "lyndsy.core.context_processors.tenant_context",  # Add tenant context
            ],
        },
    }
]

# Standardized OAuth2 refresh token cookie settings
REFRESH_COOKIE_KWARGS = dict(
    key        = "**********"
    httponly   = True,
    secure     = False,         # True in production
    samesite   = "lax",         # or "strict"
    path       = "/",           # let Vue reach it
    domain     = ".localhost.me",  # leading dot → all sub-domains
    max_age    = 60 * 60 * 24 * 30,  # 30 days
)

 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"e "**********"t "**********"_ "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"c "**********"o "**********"o "**********"k "**********"i "**********"e "**********"( "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    response.set_cookie(token, **REFRESH_COOKIE_KWARGS)

# Load OAuth2 settings from separate file
from config.settings.oauth2_settings import OAUTH2_PROVIDER, DOT_COOKIE, DOT_CLIENT_ID  # noqa

# ─────────────────────────────  3rd‑party service credentials  ─────────────────────────────

# STRIPE
# ------------------------------------------------------------------------------
STRIPE_PUBLISHABLE_KEY = env("STRIPE_PUBLISHABLE_KEY")
STRIPE_SECRET_KEY      = "**********"
STRIPE_WEBHOOK_SECRET  = "**********"
STRIPE_PRICE_ID        = env("STRIPE_PRICE_ID")
STRIPE_PRODUCT_ID      = env("STRIPE_PRODUCT_ID")

# TWILIO
# ------------------------------------------------------------------------------
TWILIO_ACCOUNT_SID            = env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN             = "**********"
TWILIO_PHONE_NUMBER           = env("TWILIO_PHONE_NUMBER")
TWILIO_MESSAGING_SERVICE_SID  = env("TWILIO_MESSAGING_SERVICE_SID")
TWILIO_WEBHOOK_URL            = env("TWILIO_WEBHOOK_URL")

# EMAIL
# ------------------------------------------------------------------------------
EMAIL_BACKEND     = "sendgrid_backend.SendgridBackend"
SENDGRID_API_KEY  = env("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = env("SENDGRID_FROM_EMAIL")

# FACEBOOK
# ------------------------------------------------------------------------------
FACEBOOK_APP_SECRET = "**********"

# CMS Blue Button 2.0 OAuth Settings
# ------------------------------------------------------------------------------
CMS_ENVIRONMENT = env("CMS_ENVIRONMENT", default="SANDBOX")
BB2_CONFIG = {
    "environment": CMS_ENVIRONMENT,
    "client_id": env("CMS_OAUTH_CLIENT_ID"),
    "client_secret": "**********"
    "callback_url": env("CMS_OAUTH_REDIRECT_URI"),
    "version": int(env("CMS_API_VERSION", "2")),
    "retry_settings": {
        "total": int(env("CMS_RETRY_TOTAL", "3")),
        "backoff_factor": int(env("CMS_RETRY_BACKOFF_FACTOR", "5")),
        "status_forcelist": [
            int(code) for code in env("CMS_RETRY_STATUS_FORCELIST", "500,502,503,504").split(",")
        ],
    },
}

USE_PKCE = env("USE_PKCE", "false").lower() in {"1", "true", "yes"}

CMS_OAUTH_CLIENT_ID     = env("CMS_OAUTH_CLIENT_ID")
CMS_OAUTH_CLIENT_SECRET = "**********"
CMS_OAUTH_REDIRECT_URI  = env("CMS_OAUTH_REDIRECT_URI")
CMS_OAUTH_SCOPES        = env("CMS_OAUTH_SCOPES", "profile,patient/Patient.read,patient/ExplanationOfBenefit.read,patient/Coverage.read").split(",")
BB2_API_URL             = env("BB2_API_URL", default="https://sandbox.bluebutton.cms.gov/v2/fhir")
CMS_OAUTH_TOKEN_URL     = env("CMS_OAUTH_TOKEN_URL", default="https: "**********"
CMS_OAUTH_AUTHORIZE_URL = env("CMS_OAUTH_AUTHORIZE_URL", default="https://sandbox.bluebutton.cms.gov/v2/o/authorize/")

CMS_OAUTH_AUTH_URL = CMS_OAUTH_AUTHORIZE_URL
CMS_OAUTH_SCOPE    = " ".join(CMS_OAUTH_SCOPES)

# Encryption Key for Tokens
# ------------------------------------------------------------------------------
TOKEN_ENCRYPTION_KEY = "**********"
# Ensure the key is bytes, as Fernet expects
 "**********"i "**********"f "**********"  "**********"i "**********"s "**********"i "**********"n "**********"s "**********"t "**********"a "**********"n "**********"c "**********"e "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"E "**********"N "**********"C "**********"R "**********"Y "**********"P "**********"T "**********"I "**********"O "**********"N "**********"_ "**********"K "**********"E "**********"Y "**********", "**********"  "**********"s "**********"t "**********"r "**********") "**********": "**********"
    TOKEN_ENCRYPTION_KEY = "**********"

try:
    Fernet(TOKEN_ENCRYPTION_KEY)  # Validate the key length
except ValueError as err:
    raise ImproperlyConfigured("TOKEN_ENCRYPTION_KEY must be a valid base64‑encoded 32‑byte key") from err

# reCAPTCHA
# ------------------------------------------------------------------------------
RECAPTCHA_PUBLIC_KEY = env("RECAPTCHA_PUBLIC_KEY", default="dummy_key")

# UHC OAuth settings
# ------------------------------------------------------------------------------
UHC_OAUTH_AUTHORIZE_URL   = env("UHC_OAUTH_AUTHORIZE_URL")
UHC_OAUTH_TOKEN_URL       = "**********"
UHC_FHIR_API_URL          = env("UHC_FHIR_API_URL")
UHC_OAUTH_CLIENT_ID       = env("UNITED_HEALTH_API_KEY")
UHC_OAUTH_CLIENT_SECRET   = "**********"
UHC_OAUTH_REDIRECT_URI    = "https://informedpluschoice.com/uhc"

# Marketplace
# ------------------------------------------------------------------------------
MARKETPLACE_BASE_URL = env("MARKETPLACE_BASE_URL")
MARKETPLACE_API_KEY  = env("MARKETPLACE_API_KEY")

# BCDA
# ------------------------------------------------------------------------------
BCDA_CLIENT_ID     = env("BCDA_CLIENT_ID")
BCDA_CLIENT_SECRET = "**********"
BCDA_API_URL       = "https://api.bcda.cms.gov"
# static/media, email, celery, logging remain mostly unchanged … (omitted for brevity)

print(f"Base settings loaded (env={ENVIRONMENT}, debug={DEBUG})")
