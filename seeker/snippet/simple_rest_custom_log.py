#date: 2023-06-21T16:52:06Z
#url: https://api.github.com/gists/0e18690f37e9e0a2161a6eb0b94f1cd1
#owner: https://api.github.com/users/saswata-dutta

from bottle import Bottle, get, post, run, request, response
from datetime import datetime
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger("app")

# set up the logger
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler("app.log", maxBytes=2**20, backupCount=5)
formatter = logging.Formatter("%(msg)s")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_to_logger(fn):
    """
    Wrap a Bottle request so that a log line is emitted after it's handled.
    (This decorator can be extended to take the desired logger as a param.)
    """

    @wraps(fn)
    def _log_to_logger(*args, **kwargs):
        request_time = datetime.now()
        actual_response = fn(*args, **kwargs)
        # modify this to log exactly what you need:
        logger.info(
            "[invoke] %s %s %s %s %s"
            % (
                request.remote_addr,
                request_time,
                request.method,
                request.url,
                response.status,
            )
        )
        return actual_response

    return _log_to_logger


app = Bottle()
app.install(log_to_logger)


@app.get("/")
def index():
    logger.info("hello in get")
    return ["hello", "world"]


@app.post("/data")
def data():
    logger.info("hello in post")
    return {"data": request.json}


app.run(host="localhost", port=8080)
