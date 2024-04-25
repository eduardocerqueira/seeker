#date: 2024-04-25T16:50:44Z
#url: https://api.github.com/gists/3e40cdbdbdd9592f1b7fffb70a934ef4
#owner: https://api.github.com/users/aleury

import os
import time

# This line needs to be run before any `ddtrace` import, to avoid sending traces
# in local dev environment (we don't have a Datadog agent configured locally, so
# it prints a stacktrace every time it tries to send a trace)
# TODO: Find a better way to activate Datadog traces?
os.environ["DD_TRACE_ENABLED"] = os.getenv("DD_TRACE_ENABLED", "false")  # noqa

import structlog
import uvicorn
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.context import correlation_id
from ddtrace.contrib.asgi.middleware import TraceMiddleware
from fastapi import FastAPI, Request, Response
from pydantic import parse_obj_as
from uvicorn.protocols.utils import get_path_with_query_string

from custom_logging import setup_logging


LOG_JSON_FORMAT = parse_obj_as(bool, os.getenv("LOG_JSON_FORMAT", False))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(json_logs=LOG_JSON_FORMAT, log_level=LOG_LEVEL)

access_logger = structlog.stdlib.get_logger("api.access")

app = FastAPI(title="Example API", version="1.0.0")


@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    structlog.contextvars.clear_contextvars()
    # These context vars will be added to all log entries emitted during the request
    request_id = correlation_id.get()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start_time = time.perf_counter_ns()
    # If the call_next raises an error, we still want to return our own 500 response,
    # so we can add headers to it (process time, request ID...)
    response = Response(status_code=500)
    try:
        response = await call_next(request)
    except Exception:
        # TODO: Validate that we don't swallow exceptions (unit test?)
        structlog.stdlib.get_logger("api.error").exception("Uncaught exception")
        raise
    finally:
        process_time = time.perf_counter_ns() - start_time
        status_code = response.status_code
        url = get_path_with_query_string(request.scope)
        client_host = request.client.host
        client_port = request.client.port
        http_method = request.method
        http_version = request.scope["http_version"]
        # Recreate the Uvicorn access log format, but add all parameters as structured information
        access_logger.info(
            f"""{client_host}:{client_port} - "{http_method} {url} HTTP/{http_version}" {status_code}""",
            http={
                "url": str(request.url),
                "status_code": status_code,
                "method": http_method,
                "request_id": request_id,
                "version": http_version,
            },
            network={"client": {"ip": client_host, "port": client_port}},
            duration=process_time,
        )
        response.headers["X-Process-Time"] = str(process_time / 10 ** 9)
        return response


# This middleware must be placed after the logging, to populate the context with the request ID
# NOTE: Why last??
# Answer: middlewares are applied in the reverse order of when they are added (you can verify this
# by debugging `app.middleware_stack` and recursively drilling down the `app` property).
app.add_middleware(CorrelationIdMiddleware)

# UGLY HACK
# Datadog's `TraceMiddleware` is applied as the very first middleware in the list, by patching `FastAPI` constructor.
# Unfortunately that means that it is the innermost middleware, so the trace/span are created last in the middleware
# chain. Because we want to add the trace_id/span_id in the access log, we need to extract it from the middleware list,
# put it back as the outermost middleware, and rebuild the middleware stack.
# TODO: Open an issue in dd-trace-py to ask if it can change its initialization, or if there is an easy way to add the
#       middleware manually, so we can add it later in the chain and have it be the outermost middleware.
# TODO: Open an issue in Starlette to better explain the order of middlewares
tracing_middleware = next(
    (m for m in app.user_middleware if m.cls == TraceMiddleware), None
)
if tracing_middleware is not None:
    app.user_middleware = [m for m in app.user_middleware if m.cls != TraceMiddleware]
    structlog.stdlib.get_logger("api.datadog_patch").info(
        "Patching Datadog tracing middleware to be the outermost middleware..."
    )
    app.user_middleware.insert(0, tracing_middleware)
    app.middleware_stack = app.build_middleware_stack()
    
    
@app.get("/")
def hello():
    custom_structlog_logger = structlog.stdlib.get_logger("my.structlog.logger")
    custom_structlog_logger.info("This is an info message from Structlog")
    custom_structlog_logger.warning("This is a warning message from Structlog, with attributes", an_extra="attribute")
    custom_structlog_logger.error("This is an error message from Structlog")

    custom_logging_logger = logging.getLogger("my.logging.logger")
    custom_logging_logger.info("This is an info message from standard logger")
    custom_logging_logger.warning("This is a warning message from standard logger, with attributes", extra={"another_extra": "attribute"})

    return "Hello, World!"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)