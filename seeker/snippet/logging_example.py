#date: 2022-02-11T16:53:55Z
#url: https://api.github.com/gists/f8a3bcfdbd587d34ead5bb4dd507029e
#owner: https://api.github.com/users/longpollehn

import sys
from pathlib import Path

def get_logger(name):
    logging.basicConfig(level=logging.DEBUG)

    # Create the logger
    logger = logging.getLogger(name)
    logger.propagate = False

    # Create handlers
    f_handler = logging.FileHandler('log.log')
    s_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler.setLevel(logging.DEBUG)
    s_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(name)s:%(lineno)d - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    s_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(f_handler)
    logger.addHandler(s_handler)

    # Return the logger
    return logger


logger = get_logger(str(Path(__file__).absolute()))
