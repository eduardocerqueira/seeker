#date: 2024-04-03T16:55:33Z
#url: https://api.github.com/gists/122b2ca4aa7577e3161a052dc9f7bd09
#owner: https://api.github.com/users/Kyu

from flask import Flask, request
import pprint


class LoggingMiddleware:
    def __init__(self, _app):
        self._app = _app

    def __call__(self, env, resp):
        error_log = env['wsgi.errors']
        pprint.pprint(('REQUEST', env), stream=error_log)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=error_log)
            return resp(status, headers, *args)

        return self._app(env, log_response)


app = Flask(__name__)
# app.wsgi_app = LoggingMiddleware(app.wsgi_app)


@app.before_request
def log_request_info():
    pprint.pprint(('HEADERS', request.headers,
                   'BODY', request.get_data()))


app.run()

""" Testing
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"username": "**********":"xyz"}' \
  http://localhost:5000
""""""