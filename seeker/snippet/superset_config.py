#date: 2024-06-24T16:49:18Z
#url: https://api.github.com/gists/08229c830407ff5e34a59aa2c8f842a2
#owner: https://api.github.com/users/youssefwadie

import logging
import os
from datetime import timedelta
from typing import Optional

from cachelib.file import FileSystemCache
from celery.schedules import crontab
from superset.superset_typing import CacheConfig

logger = logging.getLogger()

SECRET_KEY= "**********"

def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """Get the environment variable or raise exception."""
    try:
        return os.environ[var_name]
    except KeyError:
        if default is not None:
            return default
        else:
            error_msg = "The environment variable {} was missing, abort...".format(
                var_name
            )
            raise EnvironmentError(error_msg)


# DATABASE_DIALECT = get_env_variable("DATABASE_DIALECT")
# DATABASE_USER = get_env_variable("DATABASE_USER")
# DATABASE_PASSWORD = "**********"
# DATABASE_HOST = get_env_variable("DATABASE_HOST")
# DATABASE_PORT = get_env_variable("DATABASE_PORT")
# DATABASE_DB = get_env_variable("DATABASE_DB")

# # The SQLAlchemy connection string.
# SQLALCHEMY_DATABASE_URI = "%s://%s:%s@%s:%s/%s" % (
#     DATABASE_DIALECT,
#     DATABASE_USER,
#     DATABASE_PASSWORD,
#     DATABASE_HOST,
#     DATABASE_PORT,
#     DATABASE_DB,
# )

REDIS_HOST = "localhost"
REDIS_PORT = "6379"
REDIS_CELERY_DB = "0"
REDIS_RESULTS_DB = "1"

CACHE_DEFAULT_TIMEOUT = int(timedelta(days=1).total_seconds())
CACHE_CONFIG: CacheConfig = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': 'meta_',
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/2'
}
DATA_CACHE_CONFIG: CacheConfig = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': 'data_',
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/3'
}
# RESULTS_BACKEND = FileSyst  emCache("/app/superset_home/sqllab")
RESULTS_BACKEND: CacheConfig = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': 'results_',
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_RESULTS_DB}'
}
FILTER_STATE_CACHE_CONFIG: CacheConfig = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': 'filter_',
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/4'
}
EXPLORE_FORM_DATA_CACHE_CONFIG: CacheConfig = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': 'explore_',
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/5'
}
THUMBNAIL_SELENIUM_USER = "admin"
THUMBNAIL_CACHE_CONFIG: CacheConfig = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': CACHE_DEFAULT_TIMEOUT,
    'CACHE_KEY_PREFIX': 'thumbnail_',
    'CACHE_REDIS_URL': f'redis://{REDIS_HOST}:{REDIS_PORT}/6'
}


class CeleryConfig(object):
    BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_CELERY_DB}"
    CELERY_IMPORTS = ("superset.sql_lab", "superset.tasks", 'superset.tasks.thumbnails')
    CELERY_RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_RESULTS_DB}"
    CELERYD_LOG_LEVEL = "DEBUG"
    CELERYD_CONCURRENCY = 10
    CELERYD_PREFETCH_MULTIPLIER = 1
    CELERY_ACKS_LATE = False
    CELERYBEAT_SCHEDULE = {}


CELERY_CONFIG = CeleryConfig

FEATURE_FLAGS = {"THUMBNAILS": True, "THUMBNAILS_SQLA_LISTENERS": True, "ENABLE_TEMPLATE_PROCESSING": True, }
ALERT_REPORTS_NOTIFICATION_DRY_RUN = True
WEBDRIVER_TYPE = "chrome"
WEBDRIVER_BASEURL = "http://localhost:8088"
# The base URL for the email report hyperlinks.
WEBDRIVER_BASEURL_USER_FRIENDLY = WEBDRIVER_BASEURL
SUPERSET_WEBSERVER_TIMEOUT = int(timedelta(minutes=5).total_seconds())

SQLLAB_CTAS_NO_LIMIT = True

#
# Optionally import superset_config_docker.py (which will have been included on
# the PYTHONPATH) in order to allow for local settings to be overridden
#
try:
    import superset_config_docker
    from superset_config_docker import *  # noqa

    logger.info(
        f"Loaded your Docker configuration at " f"[{superset_config_docker.__file__}]"
    )
except ImportError:
    logger.info("Using default Docker config...")
except ImportError:
    logger.info("Using default Docker config...")
