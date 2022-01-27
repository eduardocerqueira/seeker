#date: 2022-01-27T17:13:46Z
#url: https://api.github.com/gists/dd6ee2f2897315c30477a0c6e16ab606
#owner: https://api.github.com/users/svozza

import boto3

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from gremlin_python import statics
from gremlin_python.process.traversal import T
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.traversal import *
from tornado import httpclient
from types import SimpleNamespace

session = boto3.Session()
credentials = session.get_credentials()
credentials = credentials.get_frozen_credentials()

def prepare_iamdb_request(database_url):
    
  service = 'neptune-db'
  method = 'GET'

  access_key = credentials.access_key
  secret_key = credentials.secret_key
  region = session.region_name
  session_token = credentials.token
  
  creds = SimpleNamespace(
    access_key=access_key, secret_key=secret_key, token=session_token, region=region,
  )

  request = AWSRequest(method=method, url=database_url, data=None)
  SigV4Auth(creds, service, region).add_auth(request)
  
  return httpclient.HTTPRequest(database_url, headers=request.headers.items())

port = 6174
server = '<my-cluster>.eu-west-1.neptune.amazonaws.com'

endpoint = f'wss://{server}:{port}/gremlin'

graph=Graph()

connection = DriverRemoteConnection(prepare_iamdb_request(endpoint), 'g')

g = graph.traversal().withRemote(connection)

data = [{"id": "1", "label": "label1", 'foo': 1,},
        {'id': '2', 'label': 'label2', 'foo': 4, 'bar': 5, 'quux': 6},
        {'id': '3', 'label': 'label3', 'foo': 7, 'bar': 8, 'quux': 9},
        {'id': '4', 'label': 'label4', 'foo': 10, 'bar': 11, 'quux': 12}]
                    
g.V('1', '2', '3', '4').drop().iterate();
                
g.inject(data).unfold().as_("nodes")\
.addV(__.select("nodes").select("label")).as_("v")\
.property(T.id, __.select("nodes").select("id"))\
.select("nodes").unfold().as_("kv")\
.select("v").property(__.select("kv").by(Column.keys), __.select("kv").by(Column.values))\
.iterate()
                
results = g.V('1', '2', '3', '4').elementMap().toList();

print(results)

connection.close()