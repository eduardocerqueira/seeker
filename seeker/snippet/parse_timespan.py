#date: 2021-09-09T17:02:02Z
#url: https://api.github.com/gists/1d28b1958b087b5947b246405fd87af0
#owner: https://api.github.com/users/benjameep

import re
def parse_timespan(str):
    SECOND = 1000
    MINUTE = SECOND*60
    HOUR = MINUTE*60
    DAY = HOUR*24
    multiplier = {
        'ms': 1,
        's':SECOND, 'sec':SECOND, 'second':SECOND, 'seconds':SECOND,
        'm':MINUTE, 'min':MINUTE, 'mins': MINUTE, 'minute': MINUTE, 'minutes':MINUTE,
        'hr':HOUR, 'hrs':HOUR, 'hour':HOUR, 'hours': HOUR,
        'd':DAY,'day':DAY,'days':DAY,
    }
    m = re.fullmatch(r'([\d\.]+)(\w+)',str.replace(' ',''))
    assert m is not None, 'is a digit followed by a word'
    assert m[2] in multiplier, 'recognized word'
    return int(float(m[1]) * multiplier[m[2]])

assert parse_timespan('25.3s') == 1000*25.3
assert parse_timespan('3 hours') == 1000*60*60*3
assert parse_timespan('1d') == 1000*60*60*24
