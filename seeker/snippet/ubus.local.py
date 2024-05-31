#date: 2024-05-31T16:58:41Z
#url: https://api.github.com/gists/1e225d7a6066976120dc10024e714587
#owner: https://api.github.com/users/lukakama

"""Action to ban IPs on an edge OpenWRT firewall.

# INSTALL
Place the file inside `/action.d` directory of fail2ban configuration directory.

# CONFIGURE
The action can be used as drop-in replacement for jails compatible with iptables actions and supports the following parameters:

- server_id : str, required
    Identifier used to create unique firewall rules names, which are composed using the following sytax:
    `f2b_<server_id>__<jail_name>__<ip>__<port>`.
    The value should be unique across all server creating rules on the same OpenWRT instance.
- url : str, required
    Root openwrt URL with procol and, optionally, port (eg: http://192.168.1.1 or  http://192.168.1.2:81)
- user : str, required
    Username used for authentication on OpenWRT.
- password : "**********"
    Password used for authentication on OpenWRT.
- protocol : str, required
    List of protocols to set like iptables action.
- port : str, required 
    List of ports and port ranges to set like iptables action.

Configurations can be defined as jail variables, like usually is done for `protocol` and `port` for jails already integrated
with iptables actions, or as action argments:

```
[jail-name]
server_id = server1
url = http://192.168.1.1
user = admin
password = "**********"
protocol = UDP,TCP
port = 80,443
action=ubus.local.py
```
or 
```
[jail-name]
action=ubus.local.py[server_id=server1, url=http: "**********"
```
or a mix
```
[jail-name]
protocol = UDP,TCP
port = 80,443
action=ubus.local.py[server_id=server1, url=http: "**********"
```

It is suggested to configure the action on the `banaction` and `banaction_allports` in the default jail configuration 
in place of the iptables ones:
```
[DEFAULT]
banaction=ubus.local.py[server_id=server1, url=http: "**********"
banaction_allports=ubus.local.py[server_id=server1, url=http: "**********"
```

"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from fail2ban.server.actions import ActionBase
from fail2ban.server.jail import Jail
import json
import re
import requests
import socket
import time
import threading

RE_CONV_RULE_NAME = re.compile(r"[^\w\d_]")
    
class UbusAction(ActionBase):
    """Fail2Ban action which configure firewall rules on OpenWRT using ubus.
    """

    def __init__(
        self, jail: "**********":str, server_id:str, url:str, user:str, password:str, protocol:str, port:str, **kwargs):
        """Initialise action."""

        super(UbusAction, self).__init__(jail, name)

        self.server_id = server_id
        self.user = user
        self.password = "**********"
        
        self.ubus_url = f"{url}/ubus"

        # Convert protocols format to OpenWRT
        self.protocols = " ".join(protocol.split(","))

        # Convert ports formats to OpenWRT for 
        self.ports = {
            ':'.join(ports_name):'-'.join([port_name if port_name.isnumeric() else str(socket.getservbyname(port_name)) for port_name in ports_name])
            for ports_name in [list(map(str.strip, ports.split(":"))) for ports in list(map(str.strip, port.split(",")))]
        }
        
    def start(self):
        """Deletes all rules relative to the current server and jail"""

        rule_name_prefix = RE_CONV_RULE_NAME.sub("_", f"f2b_{self.server_id}__{self._jail.name}__")

        ubus = "**********"
        ubus.login()
        try:
            rules = ubus.call("uci", "get", {"config": "firewall", "type": "rule"})
            for rule_name in rules["values"] if "values" in rules else []:
                if rule_name.startswith(rule_name_prefix):
                    ubus.call("uci", "delete", {"config": "firewall", "section": rule_name, "type": "rule"})

            ubus.call("uci", "commit", {"config": "firewall"})
            ubus.call("uci", "apply")
        finally:
            ubus.logout()

    def stop(self):
        """Deletes all rules relative to the current server and jail, and stop the rule watchdog"""

        rule_name_prefix = RE_CONV_RULE_NAME.sub("_", f"f2b_{self.server_id}__{self._jail.name}__")

        ubus = "**********"
        ubus.login()
        try:
            rules = ubus.call("uci", "get", {"config": "firewall", "type": "rule"})
            for rule_name in rules["values"] if "values" in rules else []:
                if rule_name.startswith(rule_name_prefix):
                    ubus.call("uci", "delete", {"config": "firewall", "section": rule_name, "type": "rule"})

            ubus.call("uci", "commit", {"config": "firewall"})
            ubus.call("uci", "apply")
        finally:
            ubus.logout()

    def ban(self, aInfo:dict):
        """Create OpenWRT firewall rules related to ports and port ranges of the ban, removing eventually expired bans."""

        ban_start_time_utc = datetime.fromtimestamp(aInfo["time"]).replace(tzinfo=timezone.utc)
        ban_end_time_utc = ban_start_time_utc + timedelta(seconds=aInfo["bantime"])

        for port_name, port_value in self.ports.items():
            rule_name = RE_CONV_RULE_NAME.sub("_", f"f2b_{self.server_id}__{self._jail.name}__{aInfo["ip"]}__{port_name}")
            rule_values = {
                "name": f"Remote fail2ban - {self.server_id} - {self._jail.name} - {aInfo["ip"]} - {port_name}",
                "proto": self.protocols,
                "src": "*",
                "src_ip": [str(aInfo["ip"])],
                "dest": "*",
                "utc_time": "1",
                "target": "DROP",
            }

            # Omit port range if all ports must be banned
            if port_value != "0-65535":
                rule_values["dest_port"] = port_value
            
            ubus = "**********"
            ubus.login()
            try:
                if ubus.call("uci", "get", {"config": "firewall", "section": rule_name}):
                    ubus.call("uci", "set",{
                        "config": "firewall",
                        "section": rule_name,
                        "type": "rule",
                        "values": rule_values,
                    })
                else:
                    ubus.call("uci", "add", {
                        "config": "firewall",
                        "name": rule_name,
                        "type": "rule",
                        "values": rule_values,
                    })

                ubus.call("uci", "commit", {"config": "firewall"})
                ubus.call("uci", "apply")
            finally:
                ubus.logout()

    def unban(self, aInfo:dict):
        """Remove all OpenWRT firewall rules related to ports and port ranges of the ban, removing eventually expired bans."""

        for port_name in self.ports:
            rule_name = RE_CONV_RULE_NAME.sub("_", f"f2b_{self.server_id}__{self._jail.name}__{aInfo["ip"]}__{port_name}")
            
            ubus = "**********"
            ubus.login()
            try:
                if ubus.call("uci", "get", {"config": "firewall", "section": rule_name}):
                    ubus.call("uci", "delete", {"config": "firewall", "section": rule_name, "type": "rule"})

                ubus.call("uci", "commit", {"config": "firewall"})
                ubus.call("uci", "apply")
            finally:
                ubus.logout()



"""Ubus session id to use for unauthenticated sessions"""
UBUS_NULL_SESSION_ID = "00000000000000000000000000000000"

"""Ubus error"""
class UbusError(Exception):
    def __init__(self, ubus_error:dict):
        self.ubus_error = ubus_error

"""JSON RPC error"""
class JsonRpcError(Exception):
    def __init__(self, json_rpc_error:dict):
        self.json_rpc_error = json_rpc_error


class Ubus():
    """Utility class for ubus invocations, handling login, session refreshes, errors and results."""
    def __init__(self, url: "**********": str, password: str):
        self.url = url
        self.user = user
        self.password = "**********"

        self.session_id = UBUS_NULL_SESSION_ID 

    def _call_raw(self, session: str, path: str, method:str, data:dict|None = None)  -> dict:
        params:list[str|dict] = [session, path, method]
        if data:
            params.append(data)

        with requests.post(self.url, 
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "call",
                    "params": params
                }) as response:
            if not response.ok:
                raise Exception(f"Ubus returned an error: {response}")
            elif 'application/json' not in response.headers.get('Content-Type', ''):
                raise Exception(f"Ubus returned a non json response: {response}")

            response_json = response.json()
            if "error" in response_json:
                raise JsonRpcError(response_json["error"])
            elif not "result" in response_json or not isinstance(response_json["result"], list) or not len(response_json["result"]) != 0:
                raise Exception(f"Ubus returned an invalid response {response}")
            elif response_json["result"][0] != 0: # UBUS_STATUS_OK
                raise UbusError(response_json["result"])
            
            return response_json["result"][1] if len(response_json["result"]) > 1 else None

    def call(self, path:str, method:str, data:dict|None = None) -> dict:
        try:
            return self._call_raw(self.session_id, path, method, data)
        except JsonRpcError as e:
            if e.json_rpc_error["code"] == -32002 and self.session_id != UBUS_NULL_SESSION_ID: 
                # Session expired, refreshing it
                self.login()
                return self._call_raw(self.session_id, path, method, data)


    def login(self) -> None:
        login_response = self._call_raw(
            UBUS_NULL_SESSION_ID,
            "session", 
            "login", 
            {
                "username": self.user,
                "password": "**********"
            }
        )
        self.session_id = login_response["ubus_rpc_session"]

    def logout(self) -> None:
        self.call("session", "destroy")
        self.session_id = UBUS_NULL_SESSION_ID

Action = UbusAction               "password": self.password
            }
        )
        self.session_id = login_response["ubus_rpc_session"]

    def logout(self) -> None:
        self.call("session", "destroy")
        self.session_id = UBUS_NULL_SESSION_ID

Action = UbusAction