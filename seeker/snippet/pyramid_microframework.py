#date: 2021-10-29T16:57:13Z
#url: https://api.github.com/gists/fc5863759460237a46eb8407e12869bb
#owner: https://api.github.com/users/matteoferla

# app -----------------------------------------------------------

from pyramid.config import Configurator
from pyramid.view import view_config
# just for typehinting...
from pyramid.request import Request
from pyramid.traversal import DefaultRootFactory
from pyramid.registry import Registry
from pyramid.response import Response
from pyramid.router import Router

def home_view(context: DefaultRootFactory, request: Request) -> dict:
    response : Respose = request.response
    registry : Registry = request.registry
    settings : dict = request.registry.settings
    return {'status': 'ok'}

with Configurator(settings=dict()) as config:
    config.add_route('home', '/')
    config.add_view(home_view, route_name='home', renderer='json')
    app  : Router = config.make_wsgi_app()
        
# run in backgroud ---------------------------------------------

port = 6969
from waitress import serve
import threading

thread = threading.Thread(target=lambda: serve(app, port=port))
thread.start()  # there is no stopping: restart kernel

# Test --------------------------------------------------------

import requests

requests.get(f'http://0.0.0.0:{port}/').json()