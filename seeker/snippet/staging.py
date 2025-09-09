#date: 2025-09-09T16:55:50Z
#url: https://api.github.com/gists/c43d6109d4525f5d90fe4781f9e41eae
#owner: https://api.github.com/users/Harshal-3558

"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import json
import os
from core.settings.base import *  # noqa
from core.utils.secret_key import generate_secret_key_if_missing
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials


cred = credentials.Certificate(os.path.join(BASE_DIR, "databrewery-db860-85d0bf9ce6ad.json"))
firebase_admin.initialize_app(cred)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(BASE_DIR, "databrewery-db860-85d0bf9ce6ad.json")
load_dotenv()

SECURE_REFERRER_POLICY = 'no-referrer-when-downgrade'
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin-allow-popups"

# SECURITY WARNING: "**********"
SECRET_KEY = "**********"

DJANGO_DB = get_env('DJANGO_DB', DJANGO_DB_POSTGRESQL)
DATABASES = {'default': DATABASES_ALL[DJANGO_DB]}

MIDDLEWARE.append('workspaces.middleware.DummyGetSessionMiddleware')
MIDDLEWARE.append('core.middleware.UpdateLastActivityMiddleware')
if INACTIVITY_SESSION_TIMEOUT_ENABLED:
    MIDDLEWARE.append('core.middleware.InactivitySessionTimeoutMiddleWare')

ADD_DEFAULT_ML_BACKENDS = False

LOGGING['root']['level'] = get_env('LOG_LEVEL', 'WARNING')

DEBUG = get_bool_env('DEBUG', True)

if DEBUG:
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
    EMAIL_FROM_ADDRESS = 'noreply@example.com'

DEBUG_PROPAGATE_EXCEPTIONS = get_bool_env('DEBUG_PROPAGATE_EXCEPTIONS', False)

SESSION_COOKIE_SECURE = get_bool_env('SESSION_COOKIE_SECURE', False)

SESSION_ENGINE = 'django.contrib.sessions.backends.signed_cookies'

# RQ_QUEUES = {}

# SENTRY_DSN = get_env('SENTRY_DSN', 'https://68b045ab408a4d32a910d339be8591a4@o227124.ingest.sentry.io/5820521')
SENTRY_ENVIRONMENT = get_env('SENTRY_ENVIRONMENT', 'opensource')

# FRONTEND_SENTRY_DSN = get_env(
#     'FRONTEND_SENTRY_DSN', 'https://5f51920ff82a4675a495870244869c6b@o227124.ingest.sentry.io/5838868'
# )
FRONTEND_SENTRY_ENVIRONMENT = get_env('FRONTEND_SENTRY_ENVIRONMENT', 'opensource')

EDITOR_KEYMAP = json.dumps(get_env('EDITOR_KEYMAP'))

from label_studio import __version__
# from label_studio.core.utils import sentry

# sentry.init_sentry(release_name='label-studio', release_version=__version__)

# we should do it after sentry init
from label_studio.core.utils.common import collect_versions

versions = collect_versions()

# in Label Studio Community version, feature flags are always ON
FEATURE_FLAGS_DEFAULT_VALUE = True
# or if file is not set, default is using offline mode
FEATURE_FLAGS_OFFLINE = get_bool_env('FEATURE_FLAGS_OFFLINE', True)

FEATURE_FLAGS_FILE = get_env('FEATURE_FLAGS_FILE', 'feature_flags.json')
FEATURE_FLAGS_FROM_FILE = True
try:
    from core.utils.io import find_node

    find_node('label_studio', FEATURE_FLAGS_FILE, 'file')
except IOError:
    FEATURE_FLAGS_FROM_FILE = False

STORAGE_PERSISTENCE = get_bool_env('STORAGE_PERSISTENCE', True)

CRONJOBS = [
    ('*/30 * * * *', 'label_studio.crons.cronjobs.nested_projects', '>> ' + os.path.join(BASE_DIR,'../crons/cronjobs.log' + ' 2>&1 '))
]

BILLING_ACCESS = ['rajdeep@databrewery.ai', 'bharat@renanpartners.com', 'pulkit@renanpartners.com', 'raaznehal225@gmail.com','shubh@renanpartners.com']

# EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
# EMAIL_HOST = 'smtp.gmail.com'  # e.g., smtp.gmail.com
# EMAIL_PORT = 587  # For TLS
# EMAIL_USE_TLS = True
# EMAIL_HOST_USER = 'noreply@databrewery.ai'
# EMAIL_HOST_PASSWORD = "**********"
# DEFAULT_FROM_EMAIL = 'noreply@databrewery.ai'

FRONTEND_HOST = 'https://staging.databrewery.ai'
# Use environment variables for security
# DEFAULT_FROM_EMAIL = 'noreply@databrewery.ai'

FRONTEND_HOST = 'https://staging.databrewery.ai'
