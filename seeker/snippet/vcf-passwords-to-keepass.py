#date: 2023-11-09T16:53:46Z
#url: https://api.github.com/gists/75a2c8a3fab483d27c25370a25bebac9
#owner: https://api.github.com/users/JuliaLblnd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lookup passwords in SDDC Manager's Password manager and store them in a KeePass database

Requirements:
	requests == 2.31.0
	pykeepass == 4.0.6

Author:
	Julia Leblond <contact@julialblnd.fr>
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from getpass import getpass
from urllib3 import disable_warnings as urllib3_disable_warnings
from requests.sessions import Session
from urllib.parse import urljoin
from json import dumps as json_dumps
from pykeepass import create_database as create_keepass

hostname = 'sddc.example.com'
username = 'sddc-lookup-user@vsphere.local'
password = "**********"
kp_password = "**********"
kdbx_file = os.path.join(os.path.dirname(__file__), 'vcf_sddc_lookup.kdbx')

# Class pour la session
# Ne pas modifier
class SDDCSession(Session):
 "**********"	 "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"h "**********"o "**********"s "**********"t "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
		super(SDDCSession, self).__init__()
		self.base_url = 'https://' + hostname
		self.headers = {'content-type': 'application/json', 'cache-control': 'no-cache'}
		self.verify = False
		# self.packages.urllib3.disable_warnings()
		urllib3_disable_warnings()

		url = 'https: "**********"
		data = {
			"username": username,
			"password": "**********"
		}
		try:
			response = self.request('POST', url, json=data)

			if response.status_code != 200:
				print(response.status_code)
				# print(response.content)
				print(json_dumps(response.json(), sort_keys=False, indent=4, separators=(',', ': ')))
				exit()

			accessToken = "**********"
			self.headers['Authorization'] = "**********"

		except Exception as e:
			raise e

	def request(self, method, service, *args, **kwargs):
		url = urljoin(self.base_url, service)
		return super(SDDCSession, self).request(method, url, *args, **kwargs)

# Lookup password with API
response = session.get('/v1/credentials?accountType=USER,SYSTEM')
# print(response.text)
data = response.json()
# print(json_dumps(data['elements'], sort_keys=True, indent=4, separators=(',', ': ')))

kp = "**********"=kp_password, keyfile=None)

for e in data['elements']:
	parent_group_name = e['accountType']
	entry_group_name  = e['resource']['resourceType']
	entry_name        = e['credentialType'] + ' - ' + e['resource']['resourceName']
	entry_username    = e['username']
	entry_password    = "**********"
	entry_url         = e['resource']['resourceName']
	entry_notes       = e.copy()
	del entry_notes['password']
	entry_notes = json_dumps(entry_notes, sort_keys=False, indent=4, separators=(',', ': '))

	# parent_group = kp.find_groups(name=parent_group_name, first=True)
	# if parent_group == None:
	# 	parent_group = kp.add_group(kp.root_group, parent_group_name)
	parent_group = kp.root_group

	entry_group = kp.find_groups(name=entry_group_name, first=True)
	if entry_group == None:
		entry_group = kp.add_group(parent_group, entry_group_name)

	kp.add_entry(entry_group, entry_name, entry_username, entry_password, url= "**********"=entry_notes)
	# kp.add_entry(destination_group, title, username, password, url= "**********"=None, tags=None, expiry_time=None, icon=None, force_creation=False)


del data['elements']
metadata_notes = json_dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
kp.add_entry(kp.root_group, "Metadata", "", "", notes=metadata_notes)

kp.save()
