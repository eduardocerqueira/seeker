#date: 2022-02-09T16:58:18Z
#url: https://api.github.com/gists/ae5753c9cd9099c02542e763052debc3
#owner: https://api.github.com/users/rudrp

import os

from flask import Flask, g

app = Flask(__name__)

class Server(object):
   def __init__(self, data):
       self.data = data

@app.route("/api_method", methods=['GET', 'POST'])
def api_method():
    return g.server.data

@app.before_request
def add_server_to_globals():
    with open(os.environ['SERVER_FILE']) as f:
        g.server = Server(f.read())

if __name__ == '__main__':
    app.run()