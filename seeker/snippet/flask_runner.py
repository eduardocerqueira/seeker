#date: 2022-09-27T17:24:33Z
#url: https://api.github.com/gists/fe5f6733fc5ab6d615d28f920b676f57
#owner: https://api.github.com/users/asehmi

import time
import datetime as dt
from flask import Flask

def flask_runner():
    app = Flask(__name__)

    @app.route('/foo')
    def serve_foo():
        ts = time.time()
        t = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        return f'Hello from Flask! ({t})'

    app.run(port=8888)

if __name__ == '__main__':
    flask_runner()
