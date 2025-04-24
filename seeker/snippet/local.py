#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

from pathlib import Path
from config.settings.base import *  # noqa: F401,F403 – import everything

# runtime ------------------------------------------------------------------------
ENVIRONMENT = "local"
DEBUG = True
ALLOWED_HOSTS += ["*.localtest.me"]

# vue dev server (via Caddy proxy)
VUE_APP_URL = "http://app.localtest.me"
LOGIN_URL = f"{VUE_APP_URL}/accounts/login"
LOGIN_REDIRECT_URL = f"{VUE_APP_URL}/app/oauth/callback"

# extra dev apps
for extra in ("debug_toolbar", "django_extensions"):
    if extra not in INSTALLED_APPS:
        INSTALLED_APPS.append(extra)

if "debug_toolbar.middleware.DebugToolbarMiddleware" not in MIDDLEWARE:
    MIDDLEWARE.insert(1, "debug_toolbar.middleware.DebugToolbarMiddleware")
INTERNAL_IPS = ["127.0.0.1"]

# dev logging --------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGING["handlers"].update(
    {
        "oauth_file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "oauth2_debug.log"),
            "formatter": "console",
            "level": "DEBUG",
        }
    }
)
LOGGING["handlers"]["console"]["level"] = "DEBUG"
for logger_name in ("django.request", "oauth2_provider", "rest_framework"):
    LOGGING["loggers"].setdefault(logger_name, {"handlers": ["console"], "level": "DEBUG", "propagate": False})


# config/settings/local.py
from cryptography.fernet import Fernet

 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"g "**********"l "**********"o "**********"b "**********"a "**********"l "**********"s "**********"( "**********") "**********". "**********"g "**********"e "**********"t "**********"( "**********"" "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"E "**********"N "**********"C "**********"R "**********"Y "**********"P "**********"T "**********"I "**********"O "**********"N "**********"_ "**********"K "**********"E "**********"Y "**********"" "**********") "**********": "**********"
    import warnings
    warnings.warn("Using insecure dev TOKEN_ENCRYPTION_KEY – DO NOT use in prod!")
    TOKEN_ENCRYPTION_KEY = "**********"
    Fernet(TOKEN_ENCRYPTION_KEY)

print("Local settings loaded – tenant switcher active on *.localtest.me")
