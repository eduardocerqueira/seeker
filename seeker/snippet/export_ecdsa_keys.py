#date: 2022-09-21T17:21:05Z
#url: https://api.github.com/gists/f0704a798af660f1ec364e28a7af200e
#owner: https://api.github.com/users/everest0407

"""This example shows how easy it is to generate and export ECDSA keys with python.

This program is similar to `ssh-keygen -t ecdsa` with no passphrase.
To export the private key with a passphrase, read paramiko.pkey.PKey._write_private_key method.
"""

import paramiko
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)

key = paramiko.ECDSAKey.generate()

with open("id_ecdsa", "wb") as fh:
    data = key.signing_key.private_bytes(Encoding.PEM,
                                         PrivateFormat.OpenSSH,
                                         NoEncryption())
    fh.write(data)

with open("id_ecdsa.pub", "wb") as fh:
    data = key.verifying_key.public_bytes(Encoding.OpenSSH,
                                          PublicFormat.OpenSSH)
    fh.write(data + b"\n")
