#date: 2024-06-24T16:56:38Z
#url: https://api.github.com/gists/892402d6bea7d309aaa4dfee3822c09e
#owner: https://api.github.com/users/thomasthaddeus

import hashlib

HASHES = ["BilboBaggins2012", "SamGamgee2013", "GandalfGrey2014"]

def print_hashes(data_list, hash_name):
    hash_func = getattr(hashlib, hash_name)
    digests = list(map(lambda data: hash_func(data.encode('utf-8')).hexdigest(), data_list))
    print(f"\n{hash_name.upper()} Hashes:")
    print("\n".join(f"{data}: {digest}" for data, digest in zip(data_list, digests)))

HASH_TYPES = ['md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512', 'blake2b', 'blake2s']

[print_hashes(HASHES, hash_name) for hash_name in HASH_TYPES]
