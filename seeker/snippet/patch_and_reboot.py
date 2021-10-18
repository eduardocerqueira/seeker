#date: 2021-10-18T17:09:37Z
#url: https://api.github.com/gists/02598a594fe609c4b8f650772a9c8790
#owner: https://api.github.com/users/alias454

import requests
import urllib3
import json
import yaml
from requests.auth import HTTPBasicAuth
from yaml.loader import SafeLoader
from time import sleep
from fabric import Connection, Config
from paramiko.ssh_exception import NoValidConnectionsError


def connect(host, user, passwd=''):
    override = Config(overrides={'sudo': {'password': passwd}})
    ret = Connection(host=host, user=user, config=override, connect_kwargs={"password": passwd})
    return ret


def cmd_wait(con, run_cmd):
    # May take up to 5 minutes
    sleep(5)
    ret = 'Command took too long to finish'
    for _ in range(25):
        try:
            ret = con.run(run_cmd, hide=True)
            break
        except (ConnectionError, NoValidConnectionsError):
            sleep(10)

    return ret


def check_status(con):
    status = con.sudo('su - splunk -c "/opt/splunk/bin/splunk status"', hide=True)
    if 'is running' in status.stdout:
        return True
    else:
        return False


# Test account credentials
username = 'admin'
password = ''
rest_user = 'admin'
rest_user_pass = ''

# Disable bad cert warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

splunk_api_url = 'https://splunk-cluster-master01.alias454.local:8089'
splunk_api_path = 'services/cluster/master/status?output_mode=json'
splunk_api = splunk_api_url + '/' + splunk_api_path

# Check if maintenance mode is NOT enabled
# False is the expected value
res = requests.get(splunk_api, auth=HTTPBasicAuth(rest_user, rest_user_pass), verify=False)
if not res.json()['entry'][0]['content']['maintenance_mode']:
    # Read inventory from file
    with open('inventory.yml') as inv_file:
        inv_data = yaml.load(inv_file, Loader=SafeLoader)

    # Setup inventory node lists
    hosts = inv_data['service']['splunk']

    # Perform checks to verify ready to start
    for role in hosts.keys():
        for hostname in hosts[role]:
            session = connect(hostname, username, password)
            if check_status(session):
                print('[OK] Splunkd is running: ' + hostname)
            else:
                print('[FAIL] Status Check: ' + hostname)
                session.close()
                exit(1)

            session.close()

    print('All checks passed... Proceeding')

    # Status checks have passed, do the updates
    for hostname in hosts['lm']:
        session = connect(hostname, username, password)
        is_rebooted = session.sudo('nohup sudo -b bash -c "sleep 5 && reboot"', hide=True)

        if is_rebooted.return_code == 0:
            res = cmd_wait(session, 'uname -a')
            print('[OK] Reboot successful: ' + hostname)
            print(res.shell)
            print('------------')

        session.close()
else:
    print('bail out since failed status checks')