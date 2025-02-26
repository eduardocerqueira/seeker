#date: 2025-02-26T16:57:31Z
#url: https://api.github.com/gists/2d90df298a017ff58ef379992aea1fa3
#owner: https://api.github.com/users/izacarias

#!/usr/bin/python3

import logging
from logging.handlers import SysLogHandler

syslogRemote = SysLogHandler(address=('127.0.0.1', 514))
remote_logger = logging.getLogger('remote_logging')
remote_logger.addHandler(syslogRemote)
remote_logger.setLevel(logging.INFO)
remote_logger.info('This is a remote log message')
remote_logger.info('It works just as the default logger ;-)')