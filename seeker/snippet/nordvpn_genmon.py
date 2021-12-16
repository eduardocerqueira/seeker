#date: 2021-12-16T17:13:13Z
#url: https://api.github.com/gists/d3d5f87abda3eae8d67ade2cfb56dd81
#owner: https://api.github.com/users/jnuxyz

#!/usr/bin/env python3

'''
# Usage

This script provides a basic (but tweakable) XFCE panel item for NordVPN, when
used in conjunction with xfce4-genmon-plugin
(https://goodies.xfce.org/projects/panel-plugins/xfce4-genmon-plugin).

The plugin also acts as a toggle button for connecting/disconnecting.

Tweaks to the command output (e.g. color) can be made based on the specs at
https://goodies.xfce.org/projects/panel-plugins/xfce4-genmon-plugin.

# Instructions

1. Install NordVPN CLI tool (https://nordvpn.com/download/)
2. Install XFCE and xfce4-genmon-plugin
3. Add an instance of Generic Monitor to the panel
4. Set the command to the following, inserting the appropriate path and country:
   `/PATH/TO/nordvpn_genmon.py --get-status --country=Canada`

## To enable auto-refresh when clicking the toggle button

4. Get the "Internal name" of the widget by hovering your mouse over it in the
   list of Panel items (e.g. "genmon-1")
5. In the Generic Monitor configuation, set the command to the following,
   inserting the appropriate path, country, and plugin name:
   `/PATH/TO/nordvpn_genmon.py --get-status --country=Canada --plugin-name=genmon-###`

# Warnings

This is an unofficial script. No guarantees can be made that this will work properly for all users. Future changes to
the NordVPN CLI tool could break parts of this script too.

Activating the Kill Switch functionality is highly recommended to prevent an instance where the NordVPN connection is
unexpectedly cut, and xfce4-genmon-plugin has not yet refreshed.
'''

import argparse
import collections
import os
import re
import subprocess

parser = argparse.ArgumentParser(description='NordVPN helper script for xfce4-genmon-plugin')
parser.add_argument('--get-status', dest='status', action='store_true', help='get formatted status message')
parser.add_argument('--connect', dest='connect', action='store_true', help='connect to NordVPN')
parser.add_argument('--disconnect', dest='disconnect', action='store_true', help='disconnect from NordVPN')
parser.add_argument('--plugin-name', dest='plugin_name', metavar='NAME', help='name of plugin in XFCE panel (e.g. "genmon-1"); enables auto-refresh after clicking toggle')
parser.add_argument('--country', dest='country', metavar='COUNTRY', help='country of server to connect to; default is "United_States"')
args = parser.parse_args()

Status = collections.namedtuple('Status', ['full_text', 'is_connected', 'ip', 'server'])


def output(msg, col='black', tooltip='', txtclick=''):
    span = '<span weight="Bold" fgcolor="{}">{}</span>'.format(col, msg)
    tool = '<tool>{}</tool>'.format(tooltip)
    tclick = '<txtclick>{}</txtclick>'.format(txtclick) if txtclick else ''
    print('<txt>{}</txt>{}{}'.format(span, tool, tclick), end='')


def get_status():
    nord = subprocess.Popen(['nordvpn', 'status'], stdout=subprocess.PIPE)
    stat = nord.stdout.read().decode().strip()
    ip = re.search(r'.*server: (.*)$', stat, re.M)
    if ip:
        ip = ip.group(1)
    server = re.search(r'.*IP: (.*)$', stat, re.M)
    if server:
        server = server.group(1)
    return Status(stat, (ip and server), ip, server)


def toggle_connection(should_connect):
    country = args.country
    nargs = ['connect', country] if should_connect else ['disconnect']
    proc = subprocess.Popen(['nordvpn'] + nargs)
    proc.wait()
    refresh_plugin()


def refresh_plugin():
    if args.plugin_name:
        proc = subprocess.Popen(['xfce4-panel', '--plugin-event={}:refresh:bool:true'.format(args.plugin_name)])
        proc.wait()


def notify(msg):
    proc = subprocess.Popen(['notify-send', msg])
    proc.wait()


def main():
    SCRIPT_PATH = os.path.realpath(__file__)
    if args.status:
        status = get_status()
        if status.ip and status.server:
            output(
                'IP: {}\nServer: {}'.format(status.ip, status.server),
                col='#008080',
                tooltip=status.full_text,
                txtclick='{} --disconnect --plugin-name={} --country={}'.format(SCRIPT_PATH, args.plugin_name, args.country)
            )
        else:
            output(
                'Not connected to NordVPN',
                col='red',
                tooltip='Click to connect to NordVPN.',
                txtclick='{} --connect --plugin-name={} --country={}'.format(SCRIPT_PATH, args.plugin_name, args.country)
            )
    elif args.connect:
        toggle_connection(True)
    elif args.disconnect:
        toggle_connection(False)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()