#date: 2023-05-18T16:45:18Z
#url: https://api.github.com/gists/fa643e563426142d9bb67539b266606d
#owner: https://api.github.com/users/arseniyGreen

import platform
import subprocess
from subprocess import Popen, PIPE
import yaml

PING_CNT = 10

def getHosts(hosts_list):
    hosts_dict = {}
    with open(hosts_list, 'r') as hosts_file:
        try:
            from_file = yaml.safe_load(hosts_file)
        except yaml.YAMLError as exc:
            print(exc)

        for host in from_file['hosts']: # parse host
            for item in host: # parse host items (ip, domain name)
                #print(host[item]['ip'])
                #print(host[item]['domain_name'])
                ip = host[item]['ip']
                dn = host[item]['domain_name']
                hosts_dict[ip] = str("💻") + dn

        return hosts_dict

def ping(host):
    param = '-n' if platform.system().lower()=='windows' else '-c'

    command = ['ping', param, str(PING_CNT), host]
    output = ''
    with Popen(command, stdout=PIPE, stderr=None) as process:
        output = process.communicate()[0].decode('UTF-8')
    return output


def print_statistics(ping_output):
    if '100% packet loss' in ping_output:
        return str("🚨No connection!")

    return str(ping_output.split('---',1)[1])

def report(report_stream):
    with open('icmp_stats.log', 'a') as f:
        f.write(report_stream)
        f.write('\n')

if __name__ == '__main__':
    hosts = getHosts('hosts.yaml')
    print(hosts)
    pings_num = "Number of pings: " + str(PING_CNT)
    report(pings_num)
    for host_ip in hosts:
        log_line = hosts[host_ip] + '\n' + print_statistics(ping(host_ip))
        report(log_line)
