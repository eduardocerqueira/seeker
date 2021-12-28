#date: 2021-12-28T17:11:04Z
#url: https://api.github.com/gists/0b22c42fd59b583bcfb77615d8f9f0b9
#owner: https://api.github.com/users/taweechai-scg

import logging
from sys import stdout

# Define logger
logger = logging.getLogger('mylogger')

logger.setLevel(logging.DEBUG) # set logger level
logFormatter = logging.Formatter\
("%(name)-12s %(asctime)s %(levelname)-8s %(filename)s:%(funcName)s %(message)s")
consoleHandler = logging.StreamHandler(stdout) #set streamhandler to stdout
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)