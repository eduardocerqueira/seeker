#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

import logging
import dj_database_url
import os



from cryptography.fernet import Fernet
from django.core.exceptions import ImproperlyConfigured

# Import env and other base settings from base.py
#   Adjust these imports as needed if your paths differ.
from .base import (
    env,
    APPS_DIR,
    BASE_DIR,
    INSTALLED_APPS,
    SPECTACULAR_SETTINGS,
    WEBPACK_LOADER,
    MIDDLEWARE, 
    TEMPLATES,
)

# Load additional environment files specific to production
env.read_env(str(BASE_DIR / ".envs" / ".production" / ".django"))
env.read_env(str(BASE_DIR / ".envs" / ".production" / ".postgres"))

# GENERAL
DEBUG = False
SECRET_KEY = "**********"
ALLOWED_HOSTS = [
    "informedpluschoice.com",
    "staging.informedpluschoice.com",
    "www.informedpluschoice.com",
    "147.135.104.192",
    "localhost",
    "127.0.0.1",
    "15.204.105.15",
    "ns1026219.ip-15-204-105.us",
]

# CORS Settings
# ------------------------------------------------------------------------------
from corsheaders.defaults import default_headers

CORS_ALLOW_CREDENTIALS = True

# Define production CORS origins
CORS_ALLOWED_ORIGINS = [
    "https://informedpluschoice.com",
    "https://www.informedpluschoice.com",
    "https://staging.informedpluschoice.com",
]

# Add localhost origins for development/testing
CORS_ALLOWED_ORIGINS += [
    "http://localhost:5173",
    "http://localhost:1313",
    "http://localhost:3000",
    "http://localhost:3001",
]

CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]

CORS_ALLOW_HEADERS = list(default_headers) + [
    "baggage",
    "sentry-trace",
    "cache-control",
    "pragma",
    "Pragma",
    "expires",
    "Expires",
]

# DATABASES
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB"),
        "USER": env("POSTGRES_USER"),
        "PASSWORD": "**********"
        "HOST": env("POSTGRES_HOST"),
        "PORT": env("POSTGRES_PORT"),
    }
}
DATABASES["default"]["ATOMIC_REQUESTS"] = True
DATABASES["default"]["CONN_MAX_AGE"] = env.int("CONN_MAX_AGE", default=60)

# Set the site ID for production
SITE_ID = 1

# https://docs.djangoproject.com/en/dev/ref/settings/#root-urlconf
ROOT_URLCONF = "config.urls"
# https://docs.djangoproject.com/en/dev/ref/settings/#wsgi-application
WSGI_APPLICATION = "config.wsgi.application"

# Redis
REDIS_URL = env("REDIS_URL", default="redis://localhost:6379/0")
# ADMIN
# ------------------------------------------------------------------------------
# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = '/data/static'
STATICFILES_DIRS = [
    '/app/static',
]



STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
]

# AUTHENTICATION
# ------------------------------------------------------------------------------
AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    
]

AUTH_USER_MODEL = "users.User"

# Login URLs
VUE_APP_URL = env("VUE_APP_URL", default="http://localhost:3000")
DJANGO_URL = env("DJANGO_URL", default="http://localhost:8000")
ACCOUNT_LOGIN_REDIRECT_URL = '/app/login'
ACCOUNT_LOGOUT_REDIRECT_URL = '/app/login'
LOGIN_URL = '/app/login'
LOGIN_REDIRECT_URL = '/app/login'


# PASSWORDS
# ------------------------------------------------------------------------------
# https: "**********"
PASSWORD_HASHERS = "**********"
    # https: "**********"
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
]
# https: "**********"
AUTH_PASSWORD_VALIDATORS = "**********"
    {
        'NAME': "**********"
    },
    {
        'NAME': "**********"
        'OPTIONS': {'min_length': 8},
    },
    {
        'NAME': "**********"
    },
    {
        'NAME': "**********"
    },
]



# Django Admin URL.
ADMIN_URL = "admin/"
# https://docs.djangoproject.com/en/dev/ref/settings/#admins
ADMINS = [("""Christian David Rodgers""", "christian@informedpluschoice.com")]
# https://docs.djangoproject.com/en/dev/ref/settings/#managers
MANAGERS = ADMINS
# https://cookiecutter-django.readthedocs.io/en/latest/settings.html#other-environment-settings

# CACHES
# ------------------------------------------------------------------------------
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": env("REDIS_URL"),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "IGNORE_EXCEPTIONS": True,  # Return empty cache on errors
        },
    }
}

# EMAIL
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#email-subject-prefix
EMAIL_SUBJECT_PREFIX = env(
    "DJANGO_EMAIL_SUBJECT_PREFIX",
    default="[Informed Choice] ",
)

EMAIL_TIMEOUT = 5

# SendGrid Email Settings
SENDGRID_API_KEY = env("SENDGRID_API_KEY", default="dummy_sendgrid_api_key")

EMAIL_BACKEND = "sendgrid_backend.SendgridBackend"
EMAIL_HOST = 'smtp.sendgrid.net'
EMAIL_HOST_USER = 'apikey'  
EMAIL_HOST_PASSWORD = "**********"
EMAIL_PORT = 587
EMAIL_USE_TLS = True

# Keep your existing DEFAULT_FROM_EMAIL setting
DEFAULT_FROM_EMAIL = 'Informed Choice Support <support@informedpluschoice.com>'
SERVER_EMAIL = DEFAULT_FROM_EMAIL
SENDGRID_SANDBOX_MODE_IN_DEBUG = False  # Set to False in production
SENDGRID_ECHO_TO_STDOUT = False  # Set to False in production

SENDGRID_FROM_EMAIL = 'Informed Choice Support <support@informedpluschoice.com>'
ADMIN_EMAIL = env("ADMIN_EMAIL", default="support@informedpluschoice.com")
# TEMPLATES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#templates
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        'DIRS': [str(APPS_DIR / "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.template.context_processors.i18n",
                "django.template.context_processors.media",
                "django.template.context_processors.static",
                "django.template.context_processors.tz",
                "django.contrib.messages.context_processors.messages",
                "lyndsy.users.context_processors.vue_app_url",
                "django.template.context_processors.csrf",
            ],
        },
    },
]


# https://docs.djangoproject.com/en/dev/ref/settings/#form-renderer
FORM_RENDERER = "django.forms.renderers.TemplatesSetting"

# http://django-crispy-forms.readthedocs.io/en/latest/install.html#template-packs
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"
CRISPY_CLASS_CONVERTERS = {
    "textinput": "tw-input",
    "fileinput": "tw-file-input",
    # ... other conversions ...
}
# FIXTURES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#fixture-dirs
FIXTURE_DIRS = (str(APPS_DIR / "fixtures"),)



# SECURITY
# ------------------------------------------------------------------------------
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 518400
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True

CSRF_COOKIE_NAME = "**********"
CSRF_COOKIE_HTTPONLY = False
CSRF_USE_SESSIONS = False
CSRF_COOKIE_SAMESITE = 'Lax'



# STRIPE
# ------------------------------------------------------------------------------
STRIPE_PUBLISHABLE_KEY = env("STRIPE_PUBLISHABLE_KEY")
STRIPE_SECRET_KEY = "**********"
STRIPE_WEBHOOK_SECRET = "**********"
STRIPE_PRICE_ID = env("STRIPE_PRICE_ID")
STRIPE_PRODUCT_ID = env("STRIPE_PRODUCT_ID")

# TWILIO
# ------------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = "**********"
TWILIO_PHONE_NUMBER = env("TWILIO_PHONE_NUMBER")
TWILIO_MESSAGING_SERVICE_SID = env("TWILIO_MESSAGING_SERVICE_SID")
TWILIO_WEBHOOK_URL = env("TWILIO_WEBHOOK_URL")
# If TWILIO_PHONE_NUMBER is repeated, ensure you keep only one consistent line

# EMAIL
# ------------------------------------------------------------------------------
EMAIL_BACKEND = "sendgrid_backend.SendgridBackend"
SENDGRID_API_KEY = env("SENDGRID_API_KEY")
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
    "version": env.int("CMS_API_VERSION", default=2),
    "retry_settings": {
        "total": env.int("CMS_RETRY_TOTAL", default=3),
        "backoff_factor": env.int("CMS_RETRY_BACKOFF_FACTOR", default=5),
        "status_forcelist": env.list(
            "CMS_RETRY_STATUS_FORCELIST",
            default=[500, 502, 503, 504],
        ),
    },
}

USE_PKCE = env.bool("USE_PKCE", default=False)

CMS_OAUTH_CLIENT_ID = env("CMS_OAUTH_CLIENT_ID")
CMS_OAUTH_CLIENT_SECRET = "**********"
CMS_OAUTH_REDIRECT_URI = env("CMS_OAUTH_REDIRECT_URI")
CMS_OAUTH_SCOPES = env.list(
    "CMS_OAUTH_SCOPES",
    default=[
        "profile",
        "patient/Patient.read",
        "patient/ExplanationOfBenefit.read",
        "patient/Coverage.read",
    ],
)
BB2_API_URL = env("BB2_API_URL", default="https://sandbox.bluebutton.cms.gov/v2/fhir")
CMS_OAUTH_TOKEN_URL = "**********"
    "CMS_OAUTH_TOKEN_URL", default="https: "**********"
)
CMS_OAUTH_AUTHORIZE_URL = env(
    "CMS_OAUTH_AUTHORIZE_URL", default="https://sandbox.bluebutton.cms.gov/v2/o/authorize/"
)

CMS_OAUTH_AUTH_URL = CMS_OAUTH_AUTHORIZE_URL
CMS_OAUTH_SCOPE = (
    "patient/Patient.read patient/Coverage.read "
    "patient/ExplanationOfBenefit.read profile"
)

# Encryption Key for Tokens
# ------------------------------------------------------------------------------
TOKEN_ENCRYPTION_KEY = "**********"
# Ensure the key is bytes, as Fernet expects
 "**********"i "**********"f "**********"  "**********"i "**********"s "**********"i "**********"n "**********"s "**********"t "**********"a "**********"n "**********"c "**********"e "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"E "**********"N "**********"C "**********"R "**********"Y "**********"P "**********"T "**********"I "**********"O "**********"N "**********"_ "**********"K "**********"E "**********"Y "**********", "**********"  "**********"s "**********"t "**********"r "**********") "**********": "**********"
    TOKEN_ENCRYPTION_KEY = "**********"

try:
    Fernet(TOKEN_ENCRYPTION_KEY)  # Validate the key length
except ValueError as err:
    msg = "**********"
    raise ImproperlyConfigured(msg) from err

# reCAPTCHA
# ------------------------------------------------------------------------------
RECAPTCHA_PUBLIC_KEY = env('RECAPTCHA_PUBLIC_KEY', default='dummy_key')

# UHC OAuth settings with default values
UHC_OAUTH_AUTHORIZE_URL = env('UHC_OAUTH_AUTHORIZE_URL')  # e.g., "https://api.uhc.com/oauth2/authorize"
UHC_OAUTH_TOKEN_URL = env('UHC_OAUTH_TOKEN_URL')          # e.g., "https: "**********"
UHC_FHIR_API_URL = env('UHC_FHIR_API_URL')                # e.g., "https://api.uhc.com/fhir"

# Map your API key/secret to the names used in the integration
UHC_OAUTH_CLIENT_ID = env('UNITED_HEALTH_API_KEY')
UHC_OAUTH_CLIENT_SECRET = "**********"

# Make sure the redirect URI is exactly as registered
UHC_OAUTH_REDIRECT_URI = "https://informedpluschoice.com/uhc"



MARKETPLACE_BASE_URL = env("MARKETPLACE_BASE_URL")
MARKETPLACE_API_KEY = env("MARKETPLACE_API_KEY")

BCDA_CLIENT_ID = env("BCDA_CLIENT_ID")
BCDA_CLIENT_SECRET = "**********"
BCDA_API_URL = "https://api.bcda.cms.gov"LIENT_ID")
BCDA_CLIENT_SECRET = env("BCDA_CLIENT_SECRET")
BCDA_API_URL = "https://api.bcda.cms.gov"