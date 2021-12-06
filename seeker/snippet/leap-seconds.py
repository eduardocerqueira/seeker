#date: 2021-12-06T16:59:20Z
#url: https://api.github.com/gists/9dc0fa6a39dafb88b9e30f514b0dfead
#owner: https://api.github.com/users/PM2Ring

# Fetch & parse leap-seconds.list
 
from datetime import datetime, timezone, timedelta
from hashlib import sha1
import requests 

# Leap second data source
# url = 'https://www.ietf.org/timezones/data/leap-seconds.list'
url = 'https://raw.githubusercontent.com/eggert/tz/main/leap-seconds.list' 

mjd_epoch = datetime(year=1900, month=1, day=1, tzinfo=timezone.utc)

def from_ntp(secs):
    return mjd_epoch + timedelta(seconds=int(secs))

req = requests.get(url)
# print(req.text)
data = req.text.splitlines()
a = [s.split() for s in data]
a = [row for row in a if len(row)>1 and len(row[0])>1]

TAI = []
lsdata = []
for row in a:
    if row[0].startswith('#'):
        c = row[0][1]
        if c=='$':
            print('Last update', from_ntp(row[1]))
            lsdata.append(row[1])
        elif c=='@':
            print('Expires', from_ntp(row[1]))
            lsdata.append(row[1])
        elif c=='h':
            hashstamp = row[1:]
            print('File hash', ' '.join(hashstamp))
        else:
            print('Bogus special comment!', repr(' '.join(row)))
    else:        
        print(' '.join(row))
        lsdata.extend(row[:2])
        TAI.append([int(u) for u in row[:2]])

# hashstamp drops leading zeroes!
hashstamp = ''.join([u.rjust(8, '0') for u in hashstamp])
print(' '*9, hashstamp)

newhash = sha1(''.join(lsdata).encode('ascii')).hexdigest()
print('Our hash ', newhash)
print('Hashes match:', hashstamp==newhash)
print()
print('TAI = [', *[f'    {row},' for row in TAI], ']', sep="\n")
