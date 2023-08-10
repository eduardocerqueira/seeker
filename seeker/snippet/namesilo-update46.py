#date: 2023-08-10T16:56:13Z
#url: https://api.github.com/gists/a34ab7f786d0abb4890483bfd321333f
#owner: https://api.github.com/users/divinehawk

#!/usr/bin/python3
# Update or add an AAAA record for your local machine using namesilo API
# Required: Set DOMAIN to your domain and KEY to your Namesilo API key
# Note: This uses Linux iproute2 for getting the IPv6 address
# Usage: ./namesilo-update46.py 
# -4 for IPv4, -6 for IPv6, or both <hostname - optional>
import requests
import xml.etree.ElementTree as ET
import socket
import os
import sys
import json
import getopt

DOMAIN = "example.com"
DEFAULT_TTL = 3600
KEY = "01234567890abcdef12345"
 
NAMESILO_LIST_RECORDS_URL = 'https://www.namesilo.com/apibatch/dnsListRecords'
NAMESILO_UPDATE_RECORD_URL = 'https://www.namesilo.com/apibatch/dnsUpdateRecord'
NAMESILO_ADD_RECORD_URL = 'https://www.namesilo.com/apibatch/dnsAddRecord'

namesilo_params = {'version': '1',
          'type': 'xml',
          'key': KEY,
          'domain': DOMAIN
}

IP4 = False 
IP6 = False
optlist, args = getopt.getopt(sys.argv[1:], "46")
for o,a in optlist:
    if o == "-4":
        IP4 = True
    elif o == "-6":
        IP6 = True
    else:
        pass # Unknown option
if len(args) == 1: 
    hostname = args[0] 
    print(hostname)
else:
    hostname = socket.gethostname()
fqdn = "{}.{}".format(hostname,DOMAIN)
print("Updating: {}".format(fqdn))

current_ip = None
if IP4:
    r = requests.get(NAMESILO_LIST_RECORDS_URL, params=namesilo_params, allow_redirects=True)
    
    xml = ET.fromstring(r.content)
    current_ip = xml.find("./request/ip").text
    code = int(xml.find("./reply/code").text)
    detail = xml.find("./reply/detail").text
    if code < 200:
        print("Error getting records: {}".format(detail))
        sys.exit(1)
    
    print('Current IPv4: {}'.format(current_ip))

current_ip_6 = None
if IP6:
    interfaces = json.loads(os.popen('ip -j addr show mngtmpaddr').read())
    for interface in interfaces:
        for addr_info in interface["addr_info"]:
            if 'family' in addr_info and addr_info['family'] == 'inet6':
                current_ip_6 = addr_info['local']
                break

    if not current_ip_6:
        print("Could not determine local IPv6 address")
        IP6 = False
    else:
        print('Current IPv6: {}'.format(current_ip_6))

if not (IP4 or IP6):
    print("Nothing to do")
    sys.exit(1)

r = requests.get(NAMESILO_LIST_RECORDS_URL, params=namesilo_params, allow_redirects=True)

xml = ET.fromstring(r.content)
current_ip = xml.find("./request/ip").text
code = int(xml.find("./reply/code").text)
detail = xml.find("./reply/detail").text
if code < 200:
    print("Error getting records: {}".format(detail))
    sys.exit(1)

found_host = False
found_record = False
for record in xml.iter('resource_record'):
    host = record.find('host').text
    value = record.find('value').text
    record_id = record.find('record_id').text
    ttl = record.find('ttl').text
    rtype = record.find('type').text
    if host == fqdn:
        found_host = True
        print("Found: {},{},{},{},{}".format(host, value, record_id, ttl, rtype))
        if rtype == 'AAAA' and IP6:
            IP6 = False
            if value != current_ip_6:
                print("It needs updating from {} to {}".format(value, current_ip_6))
                update_params = namesilo_params
                update_params['rrid'] = record_id
                update_params['rrhost'] = hostname
                update_params['rrvalue'] = current_ip_6
                update_params['rrttl'] = ttl
                r = requests.get(NAMESILO_UPDATE_RECORD_URL, params=update_params)                
                xml1 = ET.fromstring(r.content)
                code = int(xml.find("./reply/code").text)
                detail = xml.find("./reply/detail").text
                print(detail)
        elif rtype == 'A':
            IP4 = False
            if value != current_ip:
                print("It needs updating from {} to {}".format(value, current_ip))
                update_params = namesilo_params
                update_params['rrid'] = record_id
                update_params['rrhost'] = hostname
                update_params['rrvalue'] = current_ip
                update_params['rrttl'] = ttl
                r = requests.get(NAMESILO_UPDATE_RECORD_URL, params=update_params)                
                xml1 = ET.fromstring(r.content)
                code = int(xml.find("./reply/code").text)
                detail = xml.find("./reply/detail").text
                print(detail)

if IP6:
    add_params = namesilo_params
    add_params['rrtype'] = 'AAAA'
    add_params['rrhost'] = hostname
    add_params['rrvalue'] = current_ip_6
    add_params['rrttl'] = DEFAULT_TTL
    print("Adding new record")
    r = requests.get(NAMESILO_ADD_RECORD_URL, params=add_params)
    xml1 = ET.fromstring(r.content)
    code = int(xml.find("./reply/code").text)
    detail = xml.find("./reply/detail").text
    print(detail)

if IP4:
    add_params = namesilo_params
    add_params['rrtype'] = 'A'
    add_params['rrhost'] = hostname
    add_params['rrvalue'] = current_ip
    add_params['rrttl'] = DEFAULT_TTL
    print("Adding new record")
    r = requests.get(NAMESILO_ADD_RECORD_URL, params=add_params)
    xml1 = ET.fromstring(r.content)
    code = int(xml.find("./reply/code").text)
    detail = xml.find("./reply/detail").text
    print(detail)
