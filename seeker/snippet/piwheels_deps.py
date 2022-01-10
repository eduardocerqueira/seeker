#date: 2022-01-10T17:03:30Z
#url: https://api.github.com/gists/96674e828f9f3a8d104c3fe4206b10fb
#owner: https://api.github.com/users/bennuttall

from pathlib import Path

from piwheels.slave.builder import Builder, Wheel

b = Builder(None, None)
w = Wheel(Path('awkward-1.0.0-cp37-cp37m-linux_armv7l.whl'))

with open('dependencies.txt', 'w') as f:
    for pkg in w.dependencies['apt']:
        f.write(pkg + '\n')
