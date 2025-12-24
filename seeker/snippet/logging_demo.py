#date: 2025-12-24T17:04:19Z
#url: https://api.github.com/gists/357789833ea9d46a0d6734ad58d0ff73
#owner: https://api.github.com/users/sgouda0412

# Download this file with:
# curl -L https://gist.githubusercontent.com/dtaivpp/a9b00957aa7d9cfe33e92aff8d50c835/raw/logging_demo.py -o logging_demo.py
import logging

#
# Named loggers
#
logger = logging.getLogger(__name__)
print(logger.name)






#
# Log Levels
#
log_levels = [
  logging.NOTSET,
  logging.DEBUG,
  logging.INFO,
  logging.WARN,
  logging.WARNING, 
  logging.ERROR,
  logging.CRITICAL
]

for level in log_levels:
  print(level)






#
# Emmitting records
#
logger.info("Emmitting Info Record")
# Here the logger named "__main__" is emmitting an info record






#
# Handlers
#
import sys
console_handle = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handle)

# Setting Handlers Level
console_handle.setLevel(logging.WARNING)

# Sending Logs
logger.info("This won't show in the console")
logger.error("This will")


# Some common/useful handlers
from logging.handlers import \
    RotatingFileHandler, \
    SysLogHandler, \
    HTTPHandler, \
    TimedRotatingFileHandler, \
    SMTPHandler






#
# Formatters 
#
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handle.setFormatter(formatter)
logger.info("Look @ my pretty log")






# ... but the log wasn't output ...


logger.setLevel(logging.DEBUG)
logger.info("Look @ my pretty log")





# Top to bottom
#
#        Logger - Emits Records at some level
#           |
#           V
#        Handlers - Recieve logs and send to specified output 
#           |
#           V
#        Formatters - Are attached to handlers and enrich the output 
#
#        *all of these filter by log level*




#
# Setting up loggers one line at a time can be confusing
#    so we can use a json logging config. 
#
import logging
import logging.config
from logging.config import dictConfig
CONFIG = '''
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "simple": {
            "format": "%(levelname)-8s - %(message)s"
        }
    },
    "filters": {
        "warnings_and_below": {
            "()" : "__main__.filter_maker",
            "level": "WARNING"
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
            "filters": ["warnings_and_below"]
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": [
            "stderr",
            "stdout",
            "file"
        ]
    }
}
'''
dictConfig(CONFIG)




#
# There is however one other problem.... 
#
#
#
#
# Loggers are blocking operations....
#

# eg:
smtp_handler = SMTPHandler("smtp.email.com", 
                           fromaddr="fake@email.com", 
                           toaddrs="otherfake@mail.com", 
                           subject="Wont work")
smtp_handler.setLevel(logging.INFO)
logger.addHandler(smtp_handler)

for record in ["Some long list"]:
  logger.info("test")
  print(record)






#
# In comes queues to save the day!
# - note setting up queueing by dictConfig is only available in Python 3.12
#
import queue
from logging.handlers import QueueHandler, QueueListener

q = queue.Queue(-1)  # no limit on size
queue_handler = QueueHandler(q)
other_stream = logging.StreamHandler(sys.stdout)
listener = QueueListener(q, other_stream)
logger.addHandler(queue_handler)
listener.start()





# Pulling it all together with OpenSearch
# 
# Download the following docker compose file: 
# curl -L https://gist.githubusercontent.com/dtaivpp/77e310917716e49d6fafa489283847ea/raw/docker-compose.yml -o docker-compose.yml
# Run with `docker compose up -d`
#
# `pip install opensearch-py`
# Then you can run with the following:
#
import logging
from datetime import datetime
from opensearchpy import OpenSearch

opensearch_client = OpenSearch(
        "https://admin:admin@localhost:9200",
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )

# Custom Handler to send to OpenSearch
class OpenSearchHandler(logging.Handler):
    """
    Custom handler to send certain logs to OpenSearch
    """
    def __init__(self):
        logging.Handler.__init__(self)
        self.opensearch_client = opensearch_client

    def __index_name_builder(self, name: str):
        """This method creates a standardized way to build index names."""
        return f"{name}-{datetime.date(datetime.now())}"

    def emit(self, record):
        """
        Sends Log to OpenSearch
        """
        created_time = datetime.fromtimestamp(record.created).isoformat()#strftime("%Y-%m-%d %H:%M:S")

        doc = {
          "message": record.msg, 
          "locator": str(record.funcName) + " : " + str(record.lineno),
          "exec_info": record.exc_info,
          "created": created_time
        }

        self.opensearch_client.index(index=self.__index_name_builder(record.name), body=doc)

# Setup Logging
os_logger = logging.getLogger("log")
os_logger.setLevel(logging.INFO)
os_handler = OpenSearchHandler()
os_handler.setLevel(logging.INFO)
os_logger.addHandler(os_handler)

os_logger.info("This is my log~!")

# Congrats now you do more logging than 99% of the population!