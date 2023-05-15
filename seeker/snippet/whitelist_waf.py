#date: 2023-05-15T16:52:33Z
#url: https://api.github.com/gists/929788f25b01b19f332c11b7ffd0fa8f
#owner: https://api.github.com/users/filipeandre

#!/usr/bin/env python
import subprocess
subprocess.run(['pip3', 'install', 'mypy_boto3_wafv2', 'boto3'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import argparse
import ipaddress
import boto3
import requests
from mypy_boto3_wafv2 import Client


SET_NAME = "Cloudfront"
SCOPE = "CLOUDFRONT"
client: Client = boto3.client('wafv2')


def parse_args():
    """ Validate the arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', required=True)
    parser.add_argument('--waf', '-w', required=True)
    parser.add_argument('--remove', '-r',  required=False, action='store_true')
    parser.add_argument('--ip', '-i', required=False)
    return parser.parse_args()


def fatal(message):
    print(message)
    exit(1)


def info(message):
    print(f"[*] {message}")


def success(message):
    print(f"[✓] {message}")


def main():
    args = parse_args()

    if not args.ip:
        args.ip = get_external_ip()

    if not is_valid_ip(args.ip):
        fatal('Invalid ip address')

    desc = f'{args.ip} - {SET_NAME} - {args.waf} - {args.env_name}'

    if args.remove:
        info(f'Removing {desc}')
        remove_ip(f'Cloudfront{args.env_name}-{args.waf}', args.ip)
        success(f'Finished removing {desc}')
    else:
        info(f'Appending {desc}')
        add_ip(f'Cloudfront{args.env_name}-{args.waf}', args.ip)
        success(f'Finished appending {desc}')


def get_external_ip() -> str:
    info('Getting public ip address from external source')
    return requests.get('https://checkip.amazonaws.com').text.strip()


def is_valid_ip(ip: str) -> True:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def add_ip(ip_set_name: str, ip: str):
    id = get_ip_set_id(ip_set_name)
    ip_set = client.get_ip_set(Name=ip_set_name, Id=id, Scope=SCOPE)
    ip_set["IPSet"]["Addresses"].append(f'{ip}/32')
    client.update_ip_set(
        Name=ip_set_name,
        Scope=SCOPE,
        Addresses=ip_set["IPSet"]["Addresses"],
        Id=id,
        LockToken= "**********"


def remove_ip(ip_set_name: str, ip: str):
    id = get_ip_set_id(ip_set_name)
    ip_set = client.get_ip_set(Name=ip_set_name, Id=id, Scope=SCOPE)

    ip_set["IPSet"]["Addresses"].remove(f'{ip}/32'),
    client.update_ip_set(
        Name=ip_set_name,
        Scope=SCOPE,
        Addresses=ip_set["IPSet"]["Addresses"],
        Id=id,
        LockToken= "**********"


def get_ip_set_id(ip_set_name: str) -> str:
    ip_sets = list(filter(
        lambda w_acl: w_acl['Name'] == ip_set_name, client.list_ip_sets(Scope=SCOPE)['IPSets']
    ))

    if len(ip_sets) != 1:
        fatal(f'Ip set count mismatch, found: {len(ip_sets)}')

    return ip_sets[0]['Id']


if __name__ == '__main__':
    main()
':
    main()
