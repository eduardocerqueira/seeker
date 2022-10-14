#date: 2022-10-14T17:05:37Z
#url: https://api.github.com/gists/fad660cb833e5f02a8660ffd9b208dd4
#owner: https://api.github.com/users/Kasztann

# These are the settings you should have for everything to work properly.
# Add these to your main settings.py file, or modify it accordingly.

# Needed for production. Avoid using '*'.
ALLOWED_HOSTS = ['your-production-domain.com']

# Needed for 'debug' to be available inside templates.
# https://docs.djangoproject.com/en/3.2/ref/templates/api/#django-template-context-processors-debug
INTERNAL_IPS = ['127.0.0.1']

# Vite App Dir: point it to the folder your vite app is in.
VITE_APP_DIR = BASE_DIR / "src"

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

# You may change these, but it's important that the dist folder is includedself.
# If it's not, collectstatic won't copy your bundle to production.

STATIC_URL = "/static/"
STATICFILES_DIRS = [
    VITE_APP_DIR / "dist",
]
STATIC_ROOT = BASE_DIR / "staticfiles"