#date: 2022-05-31T16:53:22Z
#url: https://api.github.com/gists/e9188075572145bfee3fca475b88a07f
#owner: https://api.github.com/users/dbungert

#!/usr/bin/python3

import socket
import time

# expanded version of ipv4/ipv6 probe

# on most networks
# address 8.8.8.8 OK
# 	sock_type DGRAM
# address 8.8.8.8 OK
# 	sock_type STREAM
# address 2001:4860:4860::8888 not OK
# 	sock_type DGRAM
# 	time 2.1457672119140625e-05
# 	[Errno 101] Network is unreachable
# address 2001:4860:4860::8888 not OK
# 	sock_type STREAM
# 	time 3.814697265625e-06
# 	[Errno 101] Network is unreachable
# query_addresses ['8.8.8.8']

# on the affected network
# address 8.8.8.8 OK
#         sock_type DGRAM
# address 8.8.8.8 OK
#         sock_type STREAM
# address 2001:4860:4860::8888 OK
#         sock_type DGRAM
# address 2001:4860:4860::8888 not OK
#         sock_type STREAM
#         time 1.0243566036224365
#         [Errno 101] Network is unreachable
# query_addresses ['8.8.8.8']

check_types = {"DGRAM": socket.SOCK_DGRAM, "STREAM": socket.SOCK_STREAM}
query_addresses = []
for (af, address) in ((socket.AF_INET, '8.8.8.8'),
                      (socket.AF_INET6, '2001:4860:4860::8888')):
    has_failed = False
    for st_name, sock_type in check_types.items():
        with socket.socket(af, sock_type) as s:
            # Connecting a UDP socket is supposed to return ENETUNREACH if
            # no route to the network is present.
            start_time = time.time()
            try:
                s.connect((address, 53))
            except Exception as ex:
                elapsed = time.time() - start_time
                has_failed = True
                print(f'address {address} not OK')
                print(f'\tsock_type {st_name}')
                print(f'\ttime {elapsed}')
                print(f'\t{ex}')
            else:
                print(f'address {address} OK')
                print(f'\tsock_type {st_name}')
    if not has_failed:
        query_addresses.append(address)

print(f'query_addresses {query_addresses}')
