#date: 2022-10-19T17:29:34Z
#url: https://api.github.com/gists/8a934c5d7eef676cc4d6b4934bec9839
#owner: https://api.github.com/users/emoss08

# Prints SQL statements to console. Put this in your settings.py

LOGGING = {
    'version': 1,
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
        }
    },
    'loggers': {
        'django.db.backends': {
            'level': 'DEBUG',
            'handlers': ['console'],
        },
    }
}
SHELL_PLUS_PRINT_SQL = True