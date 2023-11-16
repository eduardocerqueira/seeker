#date: 2023-11-16T17:10:20Z
#url: https://api.github.com/gists/97efded0289456ad29a3bffa86e7dd2f
#owner: https://api.github.com/users/happygrizzly

LOGGING = {
  'version': 1,
  'disable_existing_loggers': False,
  'formatters': {
    'default': {
      'format': "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
      'datefmt': "%d/%b/%Y %H:%M:%S",
    },
    'sql': {
      'format': "[%(asctime)s] %(levelname)s [%(module)s:%(lineno)s] %(message)s",
      'datefmt': "%d/%b/%Y %H:%M:%S"
    },
  },
  'handlers': {
    'file': {
      'class': 'logging.handlers.RotatingFileHandler',
      'filename': os.path.join(os.getenv('DJANGO_LOG_PATH', BASE_DIR), 'django.default.log'),
      'maxBytes': 10 * 1024 * 1024,
      'backupCount': 5,
      'formatter': 'default',
    },
    'sql': {
      'class': 'logging.handlers.RotatingFileHandler',
      'filename': os.path.join(os.getenv('DJANGO_LOG_PATH', BASE_DIR), 'django.sql.log'),
      'maxBytes': 10 * 1024 * 1024,
      'backupCount': 5,
      'level': 'DEBUG',
      'formatter': 'sql',
    },
  },
  'loggers': {
    'django.db.backends': {
      'handlers': ['sql'],
      'level': 'DEBUG',
      'propagate': False,
    },
  },
  'root': {
    'handlers': ['file'],
    'level': os.getenv('DJANGO_LOG_LEVEL', 'WARNING'),
    'propagate': False,
  },
}