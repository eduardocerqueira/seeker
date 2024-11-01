#date: 2024-11-01T16:56:59Z
#url: https://api.github.com/gists/10f9a87046b87b8947727f487bd517b0
#owner: https://api.github.com/users/Memexurer

#!/usr/bin/env python3
import json
import hashlib
import requests
import logging
from subprocess import check_output
from datetime import datetime

class WGServer():
    def name(self):
        pass 

    def generate_config(self):
        pass

class WGServerProvider:
    def name(self):
        pass
    
    def auth(self):
        pass

    def fetch_servers(self):
        pass

logger = logging.getLogger(__name__)
key = "952b4412f002315aa50751032fcaab03"

def cauth_hash():
    epoch = int(datetime.now().timestamp())
    return hashlib.md5(f"{key}{epoch}".encode()).hexdigest()

class Windscribe(WGServerProvider):
    class WindscribeConfig(WGServer):
        def __init__(self, parent, name, host_ip, host_pubkey):
            self.parent = parent

            self.name = name
            self.host_ip = host_ip
            self.host_pubkey = host_pubkey

        def name(self):
            return self.name

        def generate_config(self):
            cfg = requests.post(
                "https://api.windscribe.net/WgConfigs/connect",
                data={
                    **self.parent.config_params,
                    "wg_pubkey": self.parent.pub,
                    "wg_ttl": "3600",
                    "hostname": self.host_ip,
                },
                headers={"User-Agent": self.parent.user_agent},
            ).json()["data"]["config"]

            return f"""
[Interface]
PrivateKey = {self.parent.priv}
Address = {cfg["Address"]}
DNS = {cfg["DNS"]}

[Peer]
PublicKey = {self.host_pubkey}
PresharedKey = {self.parent.psk}
PersistentKeepalive = 25
AllowedIPs = 200.0.0.0/5, 172.64.0.0/10, 172.128.0.0/9, 12.0.0.0/6, 16.0.0.0/4, 10.255.255.0/24, 11.0.0.0/8, 32.0.0.0/3, 128.0.0.0/3, 196.0.0.0/6, 64.0.0.0/2, 172.0.0.0/12, 194.0.0.0/7, ::/0, 192.160.0.0/13, 192.0.0.0/9, 192.170.0.0/15, 160.0.0.0/5, 192.128.0.0/11, 193.0.0.0/8, 208.0.0.0/4, 192.172.0.0/14, 176.0.0.0/4, 192.169.0.0/16, 0.0.0.0/5, 174.0.0.0/7, 192.176.0.0/12, 192.192.0.0/10, 8.0.0.0/7, 172.32.0.0/11, 173.0.0.0/8, 168.0.0.0/6, 10.255.255.2/32
Endpoint = {self.host_ip}:443
            """

    def __init__(self, user_agent="okhttp/4.10.0"):
        self.user_agent = user_agent

    def name(self):
        return "windscribe-eset"
    
    def update_session(self, session):
        self.session = session

        epoch = int(datetime.now().timestamp())

        self.client_auth_hash = cauth_hash()
        self.config_params = {
            **self.session,
            "time": epoch,
            "client_auth_hash": self.client_auth_hash,
        }

    def _existing_wg_keys(self):
        try:
            keys = json.loads(open("wireguard_keys.json").read())
            return keys["priv"], keys["pub"], keys["psk"]
        except:
            return [None, None, None]

    def _gen_wg_keys(self):
        priv = check_output("wg genkey").decode().strip()
        pub = check_output("wg pubkey", input=priv.encode()).decode().strip()
        return priv, pub

    def _save_wg_keys(self, priv, pub, psk):
        with open("wireguard_keys.json", "w") as file:
            file.write(json.dumps({"priv": priv, "pub": pub, "psk": psk}))

    def _auth_psk(self):
        self.priv, self.pub, self.psk = self._existing_wg_keys()
        if self.priv is None:
            self.priv, self.pub = self._gen_wg_keys()
            print("generated new wgpubkey: " + self.pub)

            psk = requests.post(
                "https://api.windscribe.net/WgConfigs/init",
                data={
                    **self.config_params,
                    "wg_pubkey": self.pub,
                    "wg_ttl": "3600",
                    "force_init": 1,
                },
                headers={"User-Agent": self.user_agent},
            ).json()
            self.psk = psk["data"]["config"]["PresharedKey"]

            self._save_wg_keys(self.priv, self.pub, self.psk)

    def auth(self, session_data):
        self.update_session(session_data)
        self._auth_psk()

    def register(self, license_code):
        info = {
            "platform": "android",
            "app_version": "1.2.0",
            "integration": "eset",
        }

        response = requests.post("https://api.windscribe.com/Session", json={
            **info,
            "client_auth_hash": cauth_hash(),
            "session_type_id": "4",
            "time": int(datetime.now().timestamp()),
            "license_code": license_code
        }, headers={"User-Agent": self.user_agent})

        if response.status_code != 200:
            logger.debug(response.text)
            raise Exception("license registration failed")

        session = {
            "session_auth_hash": response["session_auth_hash"],
            "license_code": license_code,
            **info,
        }
        self.auth(session)
        
        return session

    def fetch_servers(self):
        servers = []
        reqServers = requests.get(
            "https://assets.windscribe.com/serverlist/mob-v2/1/{}".format(
                self.client_auth_hash
            ),
            headers={"User-Agent": self.user_agent},
        )

        jsonServers = json.loads(reqServers.text)["data"]
        for server in jsonServers:
            for loc in server["groups"]:
                hosts = []

                try:
                    nodes = loc["nodes"]
                except KeyError:
                    continue

                for node in nodes:
                    hosts.append(node["ip3"])  # ip3, ip2, ip

                servers.append(
                    Windscribe.WindscribeConfig(
                        self,
                        "{} {} {}".format(server["name"], loc["city"], loc["nick"]),
                        hosts[0],
                        loc["wg_pubkey"],
                    )
                )

        return servers

def main():
    windscribe = Windscribe()

    windscribe.auth(json.loads(open("session.json").read()))
    servers = windscribe.fetch_servers()

    i = 0
    hosts = {}

    for server in servers:
        hosts[i] = server
        print(str(i) + ": " + server.name)
        i += 1


    server = hosts[int(input("host: "))]
    open(server.name.replace(" ", "_") + ".conf", "w").write(server.generate_config())

if __name__ == "__main__":
    main()
