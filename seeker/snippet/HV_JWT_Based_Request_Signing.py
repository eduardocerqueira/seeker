#date: 2022-05-02T17:03:02Z
#url: https://api.github.com/gists/eb85a01f49636ead0f764832a01cfae7
#owner: https://api.github.com/users/m33ch33

#!/usr/bin/env python
# -*- coding: utf-8 -*-  
#
# This Hackvertor script shows an example for creating JWT based message signature for HTTP requests.
# The code takes as parameters: path, query, and HTTP body. Hashes them, and creates a one-time JWT 
# token that is signed with the client's RSA key. The JWT token is used for both authentication and message signing. 

import json
import time
import math
import random
import base64
from hashlib import sha256
from collections import OrderedDict
import re
import sys
import os
from java.util import Base64
from javax.crypto import Cipher
from javax.crypto.spec import IvParameterSpec
from javax.crypto.spec import SecretKeySpec
from java.security import Signature
from java.security import PrivateKey
from java.security.spec import PKCS8EncodedKeySpec
from java.security import KeyFactory

# Example of JWT token produced by the code 
# {"typ":"JWT","alg":"RS256"}.{"uri":"/v1/controller/method","nonce":2080922210380511619,"iat":1644951606,
# "exp":1644951661,"sub":"41604da2-da3f-462a-bac1-3c4c0419f40e","bodyHash":
# "12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126"}.validsignature

private_key = """
-----BEGIN PRIVATE KEY-----
-----END PRIVATE KEY-----
"""
api_key = "41604da2-da3f-462a-bac1-3c4c0419f40e"

# Default param values for testing

if not 'b_full_url' in globals():
        path = "/v1/controller/method"
# else from <@set_b_body('false')>XX<@/set_b_body> set in Burp
else:
        path = str(b_full_url)

if not 'b_body' in globals():
        body_json = ""
else:
        # note that using the below line will produce a string object different 
        # from the the input, thus just take body as string
        # body_json = json.loads(str(b_body))
        body_json = str(b_body)

timestamp = time.time()
nonce = random.getrandbits(63)
timestamp_secs = int(math.floor(timestamp))
# In case there is a need for special escaping of an input data (HTTP request body in our case) 
# before it would be signed (replicating client's behaviour)
# path = path.replace("[", "%5B")
# path = path.replace("]", "%5D")

# Here we need of OrderedDict to maintain the correct order of JSON keys
# else, the signature validation in certain cases may break, since the server may rebuild the
# payload not as it was sent
# token = OrderedDict([
#         ("uri", path),
#         ("nonce", nonce),
#         ("iat", timestamp_secs),
#         ("exp", timestamp_secs + 55), 
#         ("sub", api_key),
#         ("bodyHash", str(sha256(json.dumps(body_json).encode("utf-8"))))
#         ]
# )
# json_payload = json.dumps(token,sort_keys=False)
# json_payload = re.sub("\s+","", json_payload)

# original code from
token = {
            "uri": path,
            "nonce": nonce,
            "iat": timestamp_secs,
            "exp": timestamp_secs + 55, 
            "sub": api_key,
            # Hashing request body as a string
            "bodyHash": sha256(body_json.encode("utf-8")).hexdigest()
            # Hashing request body as serialized object
            # Serialize obj to a JSON formatted str using this conversion table. 
            # https://docs.python.org/3/library/json.html
            # "bodyHash": sha256(json.dumps(body_json).encode("utf-8")).hexdigest()
        }

# As with HTTP message body, making sure our JWT's token format will match the one expected by the server;
# here should be present any specific data transformations as they performed by the client, else server 
# may fail to verify our signature
json_payload = json.dumps(
            token, separators=(",", ":"), cls=None).encode("utf-8")
# clearing whitespaces     
json_payload = re.sub("\s+","", json_payload)

# Construction header and payload parts
algorithm = "RS256"
segments = []
header = {"typ": "JWT", "alg": "RS256"}
json_header = json.dumps(header, separators=(",", ":")).encode()
segments.append(base64.urlsafe_b64encode(bytes(json_header)).replace(b"=", b""))
segments.append(base64.urlsafe_b64encode(bytes(json_payload)).replace(b"=", b""))
jwtToSign = b".".join(segments)

# Clearing out the private key; removing header,footer and whitespace
private_key = private_key.replace("-----BEGIN PRIVATE KEY-----","")
private_key = private_key.replace("-----END PRIVATE KEY-----","")
private_key = re.sub("\s+","",private_key)

# Creating the singature
keyBytes = base64.b64decode(private_key)
keySpec = PKCS8EncodedKeySpec(keyBytes);
keyFactory = KeyFactory.getInstance("RSA");
privKey = keyFactory.generatePrivate(keySpec)
Signer = Signature.getInstance("SHA256withRSA");
Signer.initSign(privKey)
Signer.update(jwtToSign)
signatureBytes = Signer.sign()

# Appending the signature to the header and the payload
segments.append(base64.urlsafe_b64encode(signatureBytes).replace(b"=", b""))
encoded_string = b".".join(segments)

# Sending back the result to Burp
output = encoded_string.decode("utf-8")

# debug output while testing the code without Burp
# output = json.dumps(body_json).encode("utf-8")
# print("\n" + str(output))


# Burp Request will look like
# POST <@set_b_full_url('false')>/v1/controller/api/object<@/set_b_full_url> HTTP/1.1
# Host: host.com
# User-Agent: python-requests/2.26.0
# Accept-Encoding: gzip, deflate
# content-type: application/json
# Accept: */*
# Connection: close
# X-API-Key: 41604da2-da3f-462a-bac1-3c4c0419f40e
# Authorization: Bearer <@_OurScriptTag('e03a3b0a2e3adb9507b2e0e758bb0a6a')><@/_OurScriptTag>
# Content-Length: 138

# <@set_b_body('false')>{"id":"someid","status":"APPROVED","address":"address","tag":"s"}<@/set_b_body>

# result
# GET /v1/controller/api/object HTTP/1.1
# Host: host.com
# User-Agent: python-requests/2.26.0
# Accept-Encoding: gzip, deflate
# content-type: application/json
# Accept: */*
# Connection: close
# X-API-Key: 41604da2-da3f-462a-bac1-3c4c0419f40e
# Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIzZWVjODY1OS03MWJmLTg1YTQ
# tMmQxNi0yY2ZlMGE4Nzc4YmQiLCJib2R5SGFzaCI6IjQyYjcxZTBhYmY0NTliNzQ4MTNmMzAwNWRjMGIwNmUzMWNjYmI1M
# 2RmMWI1YThkZTEyZGRlYzE2Mzk0ODMxODUiLCJ1cmkiOiIvdjEvY29udHJvbGxlci9hcGkvb2JqZWN0Iiwibm9uY2UiOjQ
# xNzAyOTQ0Njg4MzM4OTY2MiwiZXhwIjoxNjQ2NDgzMjk4LCJpYXQiOjE2NDY0ODMyNDN9.oDNYpgp2XdCc5uqlcthHM5lI
# Xm59Q5razTOm90AjdSzrhZv5Xab-8IMUO6GOPdioswZO4L14byRGo-mAQzivne-EROUSm9YOQMwBh6c4MHXkQCKA-U0cre
# G7hViyoSygFHsgBaPioNCBCTLSY8Xv-f3Ftz5EIZ9kd6NwThCMt0UC-GepD2QDdSutNJrNGlF-K3juOTYj3zcLxHJgbQk2
# LuwkWz0qp7kADmWmqTJNyEZaTCm16U3sS1GoT-gE_41f8W__KLdoLH3PPtZR2f6YhDBNix4P1Fb5E8xenQXTwI_eJzRi_w
# jpLmYnrtzxNxFsy4fXnRFUyRezwHVXlqdv1PbH_FaqBuKlM2u6_MqZ4h2-Ys1NTxbHU7kGqGbGw2L_xIu5A6J__9z3Q447
# uHjaIioBNUutBh_tYxFkBlL8RtKtwC3YyHim4vbyP_-C5ZmJiETx69vHW3KqdGXetRBYYjgSRW5lMJnqjuC2xRy3Dc0kRk
# yhe241kCCh3MBXjcJDFtFmbWInrvb_h5kzrKjZEc__pbyGDAlW-NmhmdF11obi8-j_Np2y93VF9xcA3F13m38z1wpEqd1k
# GG2W9-n_zgetLV4qsTK5lgzuVhlKzDiGpiydrphz2ZdiZj3YzC2w1bwL4v_fHmqnXBNyjCnNkHzi7q2vo7W67e6SZB9zTfs
# Content-Length: 138

# {"id":"someid","status":"APPROVED","address":"address","tag":"s"}

# {"typ":"JWT","alg":"RS256"}.{"sub":"3eec8659-71bf-85a4-2d16-2cfe0a8778bd","bodyHash":
# "42b71e0abf459b74813f3005dc0b06e31ccbb53df1b5a8de12ddec1639483185","uri":
# "/v1/controller/api/object","nonce":417029446883389662,"exp":1646483298,"iat":1646483243}.
# oDNYpgp2XdCc5uqlcthHM5lIXm59Q5razTOm90AjdSzrhZv5Xab-8IMUO6GOPdioswZO4L14byRGo-mAQzivne-
# EROUSm9YOQMwBh6c4MHXkQCKA-U0creG7hViyoSygFHsgBaPioNCBCTLSY8Xv-f3Ftz5EIZ9kd6NwThCMt0UC-GepD
# 2QDdSutNJrNGlF-K3juOTYj3zcLxHJgbQk2LuwkWz0qp7kADmWmqTJNyEZaTCm16U3sS1GoT-gE_41f8W__KLdoLH3
# PPtZR2f6YhDBNix4P1Fb5E8xenQXTwI_eJzRi_wjpLmYnrtzxNxFsy4fXnRFUyRezwHVXlqdv1PbH_FaqBuKlM2u6_
# MqZ4h2-Ys1NTxbHU7kGqGbGw2L_xIu5A6J__9z3Q447uHjaIioBNUutBh_tYxFkBlL8RtKtwC3YyHim4vbyP_-C5Zm
# JiETx69vHW3KqdGXetRBYYjgSRW5lMJnqjuC2xRy3Dc0kRkyhe241kCCh3MBXjcJDFtFmbWInrvb_h5kzrKjZEc__p
# byGDAlW-NmhmdF11obi8-j_Np2y93VF9xcA3F13m38z1wpEqd1kGG2W9-n_zgetLV4qsTK5lgzuVhlKzDiGpiydrph
# z2ZdiZj3YzC2w1bwL4v_fHmqnXBNyjCnNkHzi7q2vo7W67e6SZB9zTfs