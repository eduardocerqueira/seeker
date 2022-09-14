#date: 2022-09-14T17:21:30Z
#url: https://api.github.com/gists/39772c896c140d8546e6c0f315715beb
#owner: https://api.github.com/users/georges-stephan

from pgpy.pgp import PGPSignature
from base64 import encode
import pgpy
from pgpy.constants import HashAlgorithm,PubKeyAlgorithm

TEXT_TO_SIGN='Sign me pleAze!'
PRIVATE_KEY = '''
-----BEGIN PGP PRIVATE KEY BLOCK-----

lFgEYxdbiBYJKwYBBAHaRw8BAQdAmNEcIpjIgTimhaV8JFrjMXa+mnUMo1dO/rc9
SPNmNaYAAQCxpW7UrzhGn2KqqaB4GgDhQKLA+kBME0bJk851hx3HQA/4tCFnZW9y
Z2VzIDx0aGUuZ2Vvcmdlc0BleGFtcGxlLmNvbT6ImQQTFgoAQRYhBORZXDh9xbyj
huyBcgjttawXGHKWBQJjF1uIAhsDBQkDw1r4BQsJCAcCAiICBhUKCQgLAgQWAgMB
Ah4HAheAAAoJEAjttawXGHKWXjsA/3QzM7OLFcWXXCTci84Mt1GXG/V3bJbOUn8o
LQqRtWH7AP9g81XnDeTPdcSmm8whD45qCLCz4XCwSR5aQtpEGC8uCJxdBGMXW4gS
CisGAQQBl1UBBQEBB0B68FR+O9u5v1G9GAA4mN0hSAr04tZaNZGGXsBc3NSpOAMB
CAcAAP9VGf4zipU+4WYRjD/5CZhORpDSbBvihV5RyDk3pZpxEA9DiH4EGBYKACYW
IQTkWVw4fcW8o4bsgXII7bWsFxhylgUCYxdbiAIbDAUJA8Na+AAKCRAI7bWsFxhy
lgxFAP9JSEtAgJqDuR1E/6X3fjoMRobhSo9RHXqd3/PqPhHUoAD/SLusSgVlf7sC
WUOp4y25vn1hN00icEjUONkgmDjFoQU=
=O6M1
-----END PGP PRIVATE KEY BLOCK-----
'''.lstrip() # Never put a private key in your code! This is just an example

priv_key = pgpy.PGPKey()
priv_key.parse(PRIVATE_KEY)

msg = pgpy.PGPMessage.new(TEXT_TO_SIGN)
msg |= priv_key.sign(msg,hash=HashAlgorithm.SHA512)

print(f'The signature for \'{TEXT_TO_SIGN}\' is:\r\n{str(msg)}')