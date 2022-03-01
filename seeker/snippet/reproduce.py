#date: 2022-03-01T17:10:18Z
#url: https://api.github.com/gists/ec452950c317d3684016dd3b609ebca3
#owner: https://api.github.com/users/devanubis

# Importing matplotlib BEFORE helics results in exception
import matplotlib

import helics
import json
import socket

fed = helics.helicsCreateCombinationFederateFromConfig(json.dumps({
    "name": "test_federate",
    "core_type": "zmq",
    "federates": 1,
    "broker_port": 23405,
    "broker_address": socket.gethostbyname('helics'),
}))
print("Connected to broker")
print(fed)
