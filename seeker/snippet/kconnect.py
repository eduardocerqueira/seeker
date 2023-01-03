#date: 2023-01-03T16:46:08Z
#url: https://api.github.com/gists/4886d2089f671a5075a7f00fdb027acd
#owner: https://api.github.com/users/vijayanandrp

import sys
import os
import json
import argparse

PYTHON_MAJOR_VERSION = sys.version_info.major
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '8083'
BASE_PATH = '/connectors'

if PYTHON_MAJOR_VERSION == 2:
    import httplib
else:
    import http.client as httplib

if 'KAFKA_CONNECT_REST' in os.environ:
    KAFKA_CONNECT_REST = os.environ['KAFKA_CONNECT_REST']
else:
    KAFKA_CONNECT_REST = DEFAULT_HOST + ':' + str(DEFAULT_PORT)


class ConnectError(Exception):
    def __init__(self, method, path, http_status, reason):
        self.method = method
        self.path = path
        self.http_status = http_status
        self.reason = reason


class HttpUtil:

    def __init__(self, http_connection):
        self.http_connection = http_connection

    def get(self, path):
        self.http_connection.request("GET", path)
        response = self.http_connection.getresponse()

        if response.status != 200:
            raise ConnectError(method='GET', path=path, http_status=response.status, reason=response.reason)

        return json.loads(response.read())

    def post(self, path):
        self.http_connection.request('POST', path)
        response = self.http_connection.getresponse()
        response.read()

        if response.status != 204:
            raise ConnectError(method='POST', path=path, http_status=response.status, reason=response.reason)

        return {'http_status': response.status, 'reason': response.reason, 'path': path, 'method': 'POST'}

    def put(self, path):
        self.http_connection.request('PUT', path)
        response = self.http_connection.getresponse()
        response.read()

        if response.status != 202:
            raise ConnectError(method='PUT', path=path, http_status=response.status, reason=response.reason)

        return {'http_status': response.status, 'reason': response.reason, 'path': path, 'method': 'PUT'}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Functions', dest='cmd')

    cmd_status = subparsers.add_parser('status', help='show the status')
    cmd_status.add_argument('connector_id', metavar='<connector_id>', nargs='?', help='the id of the connector')

    cmd_status = subparsers.add_parser('restart', help='restart connector')
    cmd_status.add_argument('connector_id', metavar='<connector_id>', help='the id of the connector')

    cmd_status = subparsers.add_parser('pause', help='pause connector')
    cmd_status.add_argument('connector_id', metavar='<connector_id>', help='the id of the connector')

    cmd_status = subparsers.add_parser('resume', help='resume connector')
    cmd_status.add_argument('connector_id', metavar='<connector_id>', help='the id of the connector')

    args = parser.parse_args()

    conn = httplib.HTTPConnection(KAFKA_CONNECT_REST)
    http_util = HttpUtil(conn)

    try:

        if args.cmd == 'status':

            if args.connector_id:
                status = http_util.get(BASE_PATH + '/' + args.connector_id + '/status')
                print(status['name'] + ': ' + status['connector']['state'])
                for tasks in status['tasks']:
                    print('  task ' + str(tasks['id']) + ': ' + tasks['state'])

            else:
                connectors = http_util.get(BASE_PATH)
                for connector in connectors:
                    status = http_util.get(BASE_PATH + '/' + connector + '/status')
                    print(status['name'] + ': ' + status['connector']['state'])
                    for tasks in status['tasks']:
                        print('  task ' + str(tasks['id']) + ': ' + tasks['state'])

        elif args.cmd == 'pause':
            http_util.put(BASE_PATH + '/' + args.connector_id + '/pause')

        elif args.cmd == 'resume':
            http_util.put(BASE_PATH + '/' + args.connector_id + '/resume')

        elif args.cmd == 'restart':
            resp = http_util.get(BASE_PATH + '/' + args.connector_id)
            http_util.post(BASE_PATH + '/' + args.connector_id + '/restart')

            for task in resp['tasks']:
                http_util.post(BASE_PATH + '/' + args.connector_id + '/tasks/' + str(task['task']) + '/restart')

        else:
            parser.print_help()

    except ConnectError as ex:
        print('Got error %s (%s) for request %s %s%s  ' % (ex.http_status, ex.reason,
                                                           ex.method, KAFKA_CONNECT_REST, ex.path))
        print('  command: ' + args.cmd)

        if args.connector_id:
            print('  connector_id: ' + str(args.connector_id))
