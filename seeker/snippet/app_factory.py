#date: 2022-02-09T16:58:18Z
#url: https://api.github.com/gists/ae5753c9cd9099c02542e763052debc3
#owner: https://api.github.com/users/rudrp

import os

from flask import Flask, Blueprint, g

api = Blueprint('api', __name__)

class Server(object):
   def __init__(self, data):
       self.data = data

@api.route("/api_method", methods=['GET', 'POST'])
def api_method():
    return g.server.data

def make_app():
    '''
    Application factory: http://flask.pocoo.org/docs/0.10/patterns/appfactories/
    '''
    app = Flask(__name__)
    app.register_blueprint(api)

    with open(os.environ['SERVER_FILE']) as f:
        server = Server(f.read())

    @app.before_request
    def add_server_to_globals():
        g.server = server

    return app

if __name__ == '__main__':
    app = make_app()
    app.config['DEBUG'] = True
    app.run(use_debugger=True, use_reloader=True)