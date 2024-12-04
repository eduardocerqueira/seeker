#date: 2024-12-04T16:50:44Z
#url: https://api.github.com/gists/fb4ab07058187be34f6cbf7f9fef6c5a
#owner: https://api.github.com/users/tsonntag

import logging

# basic
logging.basicConfig(level=logging.ERROR)
logging.info("This is an INFO message.")

# formatting
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# file
logging.basicConfig(
    filename='app.log',
)

# complex:
logger = logging.getLogger("my_logger")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)