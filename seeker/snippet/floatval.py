#date: 2022-09-07T17:14:50Z
#url: https://api.github.com/gists/be8720456d647ab4f5f8f6f5882a1582
#owner: https://api.github.com/users/apocalyptech

#!/usr/bin/python
# vim: set expandtab tabstop=4 shiftwidth=4:

import sys
import struct

def _report_arbitrary(in_val, label, structval, big=True):
    if big:
        endian = '>'
        end_label = 'big'
    else:
        endian = '<'
        end_label = 'ltl'
    val = struct.pack(f'{endian}{structval}', in_val)
    print('{} as {} {}: {}'.format(
        in_val,
        end_label,
        label,
        val.hex(' ').upper(),
        ))

def report_arbitrary(in_val, label, structval):
    try:
        _report_arbitrary(in_val, label, structval)
        _report_arbitrary(in_val, label, structval, False)
    except struct.error:
        pass

for arg in sys.argv[1:]:

    for func, label, code in [
            (int, 'int32', 'i'),
            (int, 'uint32', 'I'),
            (int, 'int64', 'q'),
            (int, 'uint64', 'Q'),
            (float, 'float', 'f'),
            (float, 'double', 'd'),
            ]:
        try:
            report_arbitrary(func(arg), label, code)
        except ValueError:
            pass

