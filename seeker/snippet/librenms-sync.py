#date: 2024-06-28T16:36:04Z
#url: https://api.github.com/gists/8e5661df521870da6f27d5a1be83107a
#owner: https://api.github.com/users/oskaralmlov

#!/usr/bin/env python3

# Keeps LibreNMS groups in sync.

# If service templates are in use it's expected that they target groups
# whose names start with "z-service" and automatically re-discovers hosts
# that are part of these groups, if they're added / modified by the script.
# 
# Config file is expected to have this format:
# {
#   "groups": [
#   		{
#   		  "name": "WWW DB",
#   		  "rules": {"condition": "OR",
#   			    "rules": [{"field": "devices.hostname",
#   				       "id": "devices.hostname",
#   				       "input": "text",
#   				       "operator": "regex",
#   				       "type": "string",
#   				       "value": "www-db.*"}]}
#   		},
#   		{
#   		  "name": "WWW Workers",
#   		  "rules": {"condition": "OR",
#   			    "rules": [{"field": "devices.hostname",
#   				       "id": "devices.hostname",
#   				       "input": "text",
#   				       "operator": "regex",
#   				       "type": "string",
#   				       "value": "www-worker.*"}]}
#   		}
#   ]
#}

from pprint import pprint
import sys
import json
import argparse
import functools
import collections

try:
    import requests
except ImportError:
    sys.exit('requests package is required')


SERVICE_GROUP_IDENTIFIER = 'z-service'


class DeviceGroup(collections.UserDict):
    def __init__(self, name, type, rules):
        self.name = name
        self.type = type
        self.rules = rules


class LibreNMS(requests.sessions.Session):
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"h "**********"o "**********"s "**********"t "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"a "**********"p "**********"i "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"* "**********"a "**********"r "**********"g "**********"s "**********", "**********"  "**********"* "**********"* "**********"k "**********"w "**********"a "**********"r "**********"g "**********"s "**********") "**********": "**********"
        super().__init__(*args, **kwargs)

        self.identifier = '[librenms-sync]'
        self.base_url = 'https://' + hostname + '/api/v0'
        self.headers.update({'X-Auth-Token': "**********"
        self._verify_api_token_validity()

    def _request(self, method, endpoint, *args, **kwargs):
        try:
            r = self.request(method, self.base_url + endpoint, *args, **kwargs)
            return r.json()
        except requests.exceptions.RequestException as e:
            sys.exit(e)
        except Exception as e:
            sys.exit(e)

    def _get(self, endpoint):
        return self._request('get', endpoint)

    def _post(self, endpoint, *args, **kwargs):
        return self._request('post', endpoint, *args, **kwargs)

    def _del(self, endpoint, params={}, json={}):
        return self._request('delete', endpoint, *args, **kwargs)

    def _patch(self, endpoint, params={}, json={}):
        return self._request('patch', endpoint, *args, **kwargs)

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"v "**********"e "**********"r "**********"i "**********"f "**********"y "**********"_ "**********"a "**********"p "**********"i "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"v "**********"a "**********"l "**********"i "**********"d "**********"i "**********"t "**********"y "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        r = self._get('/system')
        assert r['status'] == 'ok'
        return

    def get_devicegroups(self):
        return self._get('/devicegroups')

    def list_managed_groups(self):
        r = self.get_devicegroups()
        return [g for g in r['groups'] if g['desc'] and self.identifier in g['desc']]

    def add_devicegroup(self, name, rules):
        j = {'name': name, 'rules': json.dumps(rules), 'type': 'dynamic', 'desc': self.identifier}
        return self._post('/devicegroups', json=j)

    def delete_devicegroup(self, name):
        return self._del('/devicegroups/' + name)

    def update_devicegroup(self, name, rules):
        j = {'rules': json.dumps(rules)}
        return self._patch('/devicegroups/' + name, json=j)

    def get_devices_by_group(self, name):
        return self._get('/devicegroups/' + name)

    def rediscover_device(self, name_or_id):
        return self._get('/devices/' + str(name_or_id) + '/rediscover')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-l', '--librenms-instance', required=True)
    parser.add_argument('-t', '--librenms-api-token', required= "**********"
    return parser.parse_args()


def main():
    args = parse_args()
    
    librenms = "**********"

    with open(args.config) as config_file:
        config = json.load(config_file)
    groups_in_config = config['groups']

    groups_in_librenms = librenms.list_managed_groups()
    print('Found', len(groups_in_librenms), 'managed groups in LibreNMS', '\n') 

    groups_to_add = [g for g in groups_in_config if g['name'] not in [g['name'] for g in groups_in_librenms]]
    if groups_to_add:
        print('Found', len(groups_to_add), 'groups to ADD:')
        for group in groups_to_add:
            print('*', group['name'])
        print()

    groups_to_del = [g for g in groups_in_librenms if g['name'] not in [g['name'] for g in groups_in_config]]
    if groups_to_del:
        print('Found', len(groups_to_del), 'groups to DEL:')
        for group in groups_to_del:
            print('*', group['name'])
        print()

    groups_to_mod = []
    _groups_in_librenms_map = {g['name']: g for g in groups_in_librenms}
    for group in groups_in_config:
        group_name = group['name']
        if group_name in groups_to_del:
            continue
        try:
            for idx, rule in enumerate(group['rules']['rules']):
                if rule != _groups_in_librenms_map[group_name]['rules']['rules'][idx]:
                    groups_to_mod.append(group)
                    break
        except KeyError:
            continue
          
    if groups_to_mod:
        print('Found', len(groups_to_mod), 'groups to MOD:')
        for group in groups_to_mod:
            print('*', group['name'])
        print()

    if not groups_to_add and not groups_to_del and not groups_to_mod:
        print('Nothing to do')
        sys.exit()

    answer = input('Continue? [Y/n]: ') or 'Y'
    if answer not in ('y', 'Y'):
        print('Aborted by user')
        sys.exit()

    groups_to_rediscover = set()

    for group in groups_to_add:
        a = librenms.add_devicegroup(group['name'], group['rules'])
        groups_to_rediscover.add(group['name'])
        print('Added', group['name'])

    for group in groups_to_del:
        a = librenms.delete_devicegroup(group['name'])
        print('Deleted', group['name'])

    for group in groups_to_mod:
        a = librenms.update_devicegroup(group['name'], group['rules'])
        groups_to_rediscover.add(group['name'])
        print('Modified', group['name'])


    devices_to_rediscover = set()
    for group_name in groups_to_rediscover:
        if not group_name.startswith(SERVICE_GROUP_IDENTIFIER):
            continue
        devices = librenms.get_devices_by_group(group_name)
        devices_to_rediscover.update(set([d['device_id'] for d in devices['devices'] ]))

    if devices_to_rediscover:
        print()
        print('Re-discovering', len(devices_to_rediscover), 'devices associated with', len(groups_to_rediscover), 'service group(s):')
        for device_id in devices_to_rediscover:
            librenms.rediscover_device(device_id)
            print(device_id)



if __name__ == '__main__':
    main()
