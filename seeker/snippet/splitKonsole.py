#date: 2022-01-27T17:12:36Z
#url: https://api.github.com/gists/32c27d880a9e281d361c96fbc17c9d4a
#owner: https://api.github.com/users/TomasGlgg

from pwn import which
import os, subprocess
from bs4 import BeautifulSoup

if not 'KONSOLE_VERSION' in os.environ:
    print('Konsole not found')
    exit(1)


qdbus = which('qdbus')
window_id = os.environ['WINDOWID']
konsole_dbus_service = os.environ['KONSOLE_DBUS_SERVICE']

with subprocess.Popen((qdbus, konsole_dbus_service, '/konsole', 'org.freedesktop.DBus.Introspectable.Introspect'),
                      stdout=subprocess.PIPE) as proc:
    xml = proc.communicate()[0].decode()
    parser = BeautifulSoup(xml, 'html.parser')

# Iterate over all nodes
for MainWindow in parser.findAll('node'):
    name = MainWindow.get('name')
    if name and name.startswith('MainWindow_'):
        with subprocess.Popen((qdbus, konsole_dbus_service, '/konsole/' + name,
                               'org.kde.KMainWindow.winId'), stdout=subprocess.PIPE) as proc:
            target_window_id = proc.communicate()[0].decode().strip()
            if target_window_id == window_id:
                break
else:
    print('MainWindow not found')
    exit(1)
print('Result:', name)
