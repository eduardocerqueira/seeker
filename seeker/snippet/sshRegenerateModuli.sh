#date: 2022-05-31T17:00:29Z
#url: https://api.github.com/gists/98cffd9d40c7f2a800403dc0082a661f
#owner: https://api.github.com/users/marksharrison

#!/bin/sh
# regenerate list of prime numbers to replace pre-generated list
# marginally increases security of key exchange protocols
# this will take hours or possibly days
ssh-keygen -M generate -O bits=8192 moduli-8192.all
# verify regenerated list of primes are valid and not inherently weak
ssh-keygen -M screen -f moduli-8192.all moduli-8192
# replace the original /etc/ssh/moduli file
cp moduli-8192 /etc/ssh/moduli
rm moduli-8192