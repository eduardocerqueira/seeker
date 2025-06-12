#date: 2025-06-12T17:12:31Z
#url: https://api.github.com/gists/2be63853aa3b0c218628ce861e49c888
#owner: https://api.github.com/users/mpslanker

import logging

# Configure logging to print to the console
logging.basicConfig(level=logging.DEBUG)

# Create a logger
logger = logging.getLogger(__name__)

# Log messages at different levels
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")