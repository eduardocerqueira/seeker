#date: 2024-11-26T17:10:28Z
#url: https://api.github.com/gists/09460fff1e1b02a0ded2c1f722bbc861
#owner: https://api.github.com/users/User087

import http.client
import http.cookies
import json
import base64
import hashlib
import os
import random
import argparse
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

"""
Copyright - FuseTim 2024

This code is dual-licensed under both the MIT License and the Apache License 2.0.

You may choose either license to govern your use of this code.

MIT License:
https://opensource.org/licenses/MIT

Apache License 2.0:
https://www.apache.org/licenses/LICENSE-2.0

By contributing to this project, you agree that your contributions will be licensed under 
both the MIT License and the Apache License 2.0.
"""

######################################################################################

# Credentials (found in Headers and Cookies)
auth_server = "" # See `x-pm-uid` header
auth_token  = "**********"
session_id  = "" # See `Session-Id` cookie
web_app_version = "web-vpn-settings@5.0.2.0" # See `x-pm-appversion` header

# Settings
prefix = "Wire_AuTo"  # Prefix is used for config file and name in ProtonVPN Dashboard
output_dir = "./"
selected_countries = ["US"]
selected_cities = ["Chicago", "New York", "Ashburn"]
selected_tier = 2  # 0 = Free, 2 = Plus
selected_features = ["P2P"]  # Features that a server should have ("P2P", "TOR", "SecureCore", "XOR", etc) or not ("-P2P", etc)
max_servers = 30  # Maximum of generated config
max_servers_per_city = 10  # Maximum number of servers per city
listing_only = False  # Do not generate config, just list available servers with previous selectors

config_features = {
    "SafeMode": False,
    "SplitTCP": True,
    "PortForwarding": True,
    "RandomNAT": False,
    "NetShieldLevel": 0,  # 0, 1 or 2
}

######################################################################################

# Constants
connection = http.client.HTTPSConnection("account.protonvpn.com")
C = http.cookies.SimpleCookie()
C["AUTH-" + auth_server] = "**********"
C["Session-Id"] = session_id
headers = {
    "x-pm-appversion": web_app_version,
    "x-pm-uid": auth_server,
    "Accept": "application/vnd.protonmail.v1+json",
    "Cookie": C.output(attrs=[], header="", sep="; ")
}


def generateKeys():
    """Generate a client key-pair using the API. Could be generated offline but need more work..."""
    print("Generating key-pair...")
    connection.request("GET", "/api/vpn/v1/certificate/key/EC", headers=headers)
    response = connection.getresponse()
    print("Status: {} and reason: {}".format(response.status, response.reason))
    resp = json.loads(response.read().decode())
    priv = resp["PrivateKey"].split("\n")[1]
    pub = resp["PublicKey"].split("\n")[1]
    print("Key generated:")
    print("priv:", priv)
    print("pub:", pub)
    return [resp["PrivateKey"], pub, priv]


def getPubPEM(priv):
    """Return the Public key as string without headers"""
    return priv[1]


def getPrivPEM(priv):
    """Return the Private key as PKCS#8 without headers"""
    return priv[2]


def getPrivx25519(priv):
    """Return the x25519 base64-encoded private key, to be used in Wireguard config."""
    hash__ = hashlib.sha512(base64.b64decode(priv[2])[-32:]).digest()
    hash_ = list(hash__)[:32]
    hash_[0] &= 0xf8
    hash_[31] &= 0x7f
    hash_[31] |= 0x40
    new_priv = base64.b64encode(bytes(hash_)).decode()
    return new_priv


def registerConfig(server, priv):
    """Register a Wireguard configuration and return its raw response."""
    h = headers.copy()
    h["Content-Type"] = "application/json"
    print("Registering Config for server", server["Name"], "...")
    body = {
        "ClientPublicKey": getPubPEM(priv),
        "Mode": "persistent",
        "DeviceName": prefix + "-" + server["Name"].replace("#", "-"),
        "Features": {
            "peerName": server["Name"],
            "peerIp": server["Servers"][0]["EntryIP"],
            "peerPublicKey": server["Servers"][0]["X25519PublicKey"],
            "platform": "Windows",
            "SafeMode": config_features["SafeMode"],
            "SplitTCP": config_features["SplitTCP"],
            "PortForwarding": config_features["PortForwarding"] if server["Features"] & 4 == 4 else False,
            "RandomNAT": config_features["RandomNAT"],
            "NetShieldLevel": config_features["NetShieldLevel"],  # 0, 1 or 2
        }
    }
    print("Request body for new config:", json.dumps(body, indent=4))  # Log the request body
    connection.request("POST", "/api/vpn/v1/certificate", body=json.dumps(body), headers=h)
    response = connection.getresponse()
    print("Status: {} and reason: {}".format(response.status, response.reason))
    resp = json.loads(response.read().decode())
    print(resp)
    return resp


def generateConfig(priv, register):
    """Generate a Wireguard config using the ProtonVPN API answer."""
    conf = """[Interface]
# Key for {prefix}
PrivateKey = {priv}
Address = 10.2.0.2/32
DNS = 192.168.1.254

[Peer]
# {server_name}
PublicKey = {server_pub}
AllowedIPs = 0.0.0.0/1, 128.0.0.0/1
Endpoint = {server_endpoint}:51820
    """.format(prefix=prefix, priv=getPrivx25519(priv), server_name=register["Features"]["peerName"],
               server_pub=register["Features"]["peerPublicKey"], server_endpoint=register["Features"]["peerIp"])
    return conf


def write_config_to_disk(name, conf):
    with open(output_dir + "/" + name + ".conf", "w") as f:
        f.write(conf)


def read_conf_file(file_path):
    """Read the Wireguard configuration file and extract details."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    config_data = {}
    for line in lines:
        if line.startswith("PrivateKey = "):
            config_data["PrivateKey"] = line.split(" = ")[1].strip()
        elif line.startswith("PublicKey = "):
            config_data["PublicKey"] = line.split(" = ")[1].strip()
        elif line.startswith("Endpoint = "):
            config_data["Endpoint"] = line.split(" = ")[1].strip().split(":")[0]

    return config_data


def renew_certificate_from_conf(device_name, cert):
    """Renew a Wireguard configuration by generating a new key pair."""
    h = headers.copy()
    h["Content-Type"] = "application/json"
    print("Renewing Config for server", device_name, "...")

    new_keys = generateKeys()  # Generate a new key pair
    body = {
        "ClientPublicKey": new_keys[1],
        "Mode": "persistent",
        "DeviceName": device_name,
        "Features": {
            "peerName": cert["Features"]["peerName"],
            "peerIp": cert["Features"]["peerIp"],
            "peerPublicKey": cert["Features"]["peerPublicKey"],
            "platform": cert["Features"]["platform"],
            "SafeMode": config_features["SafeMode"],
            "SplitTCP": config_features["SplitTCP"],
            "PortForwarding": config_features["PortForwarding"],
            "RandomNAT": config_features["RandomNAT"],
            "NetShieldLevel": config_features["NetShieldLevel"],
        },
        "Renew": True
    }
    print("Request body for renewing config:", json.dumps(body, indent=4))  # Log the request body
    connection.request("POST", "/api/vpn/v1/certificate", body=json.dumps(body), headers=h)
    response = connection.getresponse()
    print("Status: {} and reason: {}".format(response.status, response.reason))
    resp = json.loads(response.read().decode())
    print(resp)
    return resp, new_keys


def parse_arguments():
    parser = argparse.ArgumentParser(description='ProtonVPN configuration script')
    parser.add_argument('-extend', action='store_true', help='Renew existing certificates')
    return parser.parse_args()


def fetch_existing_certificates():
    """Fetch existing certificates from ProtonVPN account"""
    h = headers.copy()
    h["Content-Type"] = "application/json"
    connection.request("GET", "/api/vpn/v1/certificate/all?Mode=persistent&Offset=0&Limit=51", headers=h)
    response = connection.getresponse()
    print("Status: {} and reason: {}".format(response.status, response.reason))
    resp = json.loads(response.read().decode())
    print("Fetched existing certificates:", resp)
    return resp["Certificates"]


def get_existing_config_names():
    """Get the list of existing configuration names without the file extension."""
    return [f.split(".")[0] for f in os.listdir(output_dir) if f.endswith(".conf")]


# VPN Listings
connection.request("GET", "/api/vpn/logicals", headers=headers)
response = connection.getresponse()
print("Status: {} and reason: {}".format(response.status, response.reason))

servers = json.loads(response.read().decode())["LogicalServers"]

# Create a dictionary to track the number of servers per city
servers_per_city = {city: 0 for city in selected_cities}

# Collect eligible servers first
eligible_servers = []

for s in servers:
    feat = [
        "SecureCore" if s["Features"] & 1 == 1 else "-SecureCore",
        "TOR" if s["Features"] & 2 == 2 else "-TOR",
        "P2P" if s["Features"] & 4 == 4 else "-P2P",
        "XOR" if s["Features"] & 8 == 8 else "-XOR",
        "IPv6" if s["Features"] & 16 == 16 else "-IPv6"
    ]
    if (not s["EntryCountry"] in selected_countries and not s["ExitCountry"] in selected_countries) or s["Tier"] != selected_tier:
        continue
    if s["City"] not in selected_cities:
        continue
    if len(list(filter(lambda sf: not (sf in feat), selected_features))) > 0:
        continue
    eligible_servers.append(s)

# Remove servers for which configurations already exist
existing_configs = get_existing_config_names()
eligible_servers = [s for s in eligible_servers if prefix + "-" + s["Name"].replace("#", "-") not in existing_configs]

# Shuffle the eligible servers to ensure randomness
random.shuffle(eligible_servers)

if __name__ == "__main__":
    args = parse_arguments()

    if args.extend:
        print("Renewing existing certificates...")
        existing_certificates = fetch_existing_certificates()

        for cert in existing_certificates:
            device_name = cert["DeviceName"].replace("#", "-")
            if device_name in existing_configs:
                renewed_config, new_keys = renew_certificate_from_conf(device_name, cert)
                config = generateConfig(new_keys, renewed_config)
                write_config_to_disk(device_name, config)
                time.sleep(60)
    else:
        print("Generating new configurations...")
        total_configs = sum(servers_per_city.values())

        for s in eligible_servers:
            if total_configs >= max_servers:
                break
            device_name = prefix + "-" + s["Name"].replace("#", "-")
            if servers_per_city[s["City"]] >= max_servers_per_city:
                continue

            servers_per_city[s["City"]] += 1
            total_configs += 1
            keys = generateKeys()
            reg = registerConfig(s, keys)
            config = generateConfig(keys, reg)
            write_config_to_disk(device_name, config)
            time.sleep(60)

connection.close()
connection.close()
