#date: 2024-03-08T17:02:13Z
#url: https://api.github.com/gists/1604032bd0599ede997fc3136388750d
#owner: https://api.github.com/users/katlol

# wg-fullmesh.py
# requires pip package wgconfig
# CC0 - public domain - made by github.com/katlol

import configparser
import json
import pathlib
import subprocess
import sys
from functools import partial

import wgconfig
import wgconfig.wgexec as wgexec


class WireGuardHost:
    def __init__(
        self, name: str, ip: str, wg_ip: str, path: pathlib.Path, port: int = 51820
    ):
        self.name = name
        self.ip = ip
        self.wg_ip = wg_ip
        self.path = path
        self.port = port
        self.config = wgconfig.WGConfig(path)

    def __eq__(self, other):
        our_privkey = self.config.get_interface().get("PrivateKey")
        their_privkey = other.config.get_interface().get("PrivateKey")
        return our_privkey == their_privkey

    def __hash__(self):
        return hash(self.config.get_interface().get("PrivateKey"))

    def _write(self):
        self.config.write_file()

    def _read(self):
        self.config.read_file()


PWD = pathlib.Path(sys.path[0])
CONFIG_FOLDER = PWD
Hosts = [
    WireGuardHost(
        name="example-server-a-0",
        ip="127.0.0.1",
        wg_ip="127.0.0.1",
        path=CONFIG_FOLDER / "example-server-a-0.wg.conf",
    ),
    WireGuardHost(
        name="example-server-a-1",
        ip="127.0.0.1",
        wg_ip="127.0.0.1",
        path=CONFIG_FOLDER / "example-server-a-1.wg.conf",
    ),
    WireGuardHost(
        name="example-server-a-2",
        ip="127.0.0.1",
        wg_ip="127.0.0.1",
        path=CONFIG_FOLDER / "example-server-a-2.wg.conf",
    ),
    WireGuardHost(
        name="example-server-b-0",
        ip="127.0.0.1",
        wg_ip="127.0.0.1",
        path=CONFIG_FOLDER / "example-server-b-0.wg.conf",
    ),
    WireGuardHost(
        name="example-server-b-1",
        ip="127.0.0.1",
        wg_ip="127.0.0.1",
        path=CONFIG_FOLDER / "example-server-b-1.wg.conf",
    ),
    WireGuardHost(
        name="example-server-b-2",
        ip="127.0.0.1",
        wg_ip="127.0.0.1",
        path=CONFIG_FOLDER / "example-server-b-2.wg.conf",
    ),
]

PWD = pathlib.Path(sys.path[0])


def _init():
    for host in Hosts:
        debug = partial(print, host.name, "---", sep=" " * 5, end="\n", flush=True)

        # If file does not exist, initialize it
        if not host.path.exists():
            debug("initializing")
            comment_json = json.dumps({"name": host.name, "ip": host.ip})
            host.config.initialize_file(leading_comment="#" + comment_json)
        else:
            debug("exists")
            host.config.read_file()
        interface = host.config.get_interface()
        # Ensure Address is set
        if not interface.get("Address"):
            debug("setting Address")
            host.config.add_attr(None, "Address", host.wg_ip)
        # Ensure ListenPort is set
        if not interface.get("ListenPort"):
            debug("setting ListenPort")
            host.config.add_attr(None, "ListenPort", host.port)
        # Ensure PrivateKey is set
        if not interface.get("PrivateKey"):
            debug("setting PrivateKey")
            host.config.add_attr(None, "PrivateKey", wgexec.generate_privatekey())
        debug("writing")
        host.config.write_file()


def _reconcile_peers():
    for host in Hosts:
        debug = partial(print, host.name, "---", sep=" " * 5, end="\n", flush=True)
        for peer in Hosts:
            host._read()
            peer._read()
            if host == peer:
                continue
            debug("peer", peer.name)
            # Reconcile
            host_if = host.config.get_interface()
            peer_if = peer.config.get_interface()
            host_pubkey = wgexec.get_publickey(host_if["PrivateKey"])
            peer_pubkey = wgexec.get_publickey(peer_if["PrivateKey"])

            # 0. peer exists in host
            try:
                peer_section = host.config.get_peer(peer_pubkey)
                debug("peer exists", peer.name)
            except KeyError:
                debug("peer does not exist", peer.name)
                comment_json = json.dumps({"name": peer.name, "ip": peer.ip})
                host.config.add_peer(peer_pubkey, leading_comment="#" + comment_json)
                peer_section = host.config.get_peer(peer_pubkey)

            # important, other way around too,
            try:
                host_section = peer.config.get_peer(host_pubkey)
                debug("host exists", host.name)
            except KeyError:
                debug("host does not exist", host.name)
                comment_json = json.dumps({"name": host.name, "ip": host.ip})
                peer.config.add_peer(host_pubkey, leading_comment="#" + comment_json)
                host_section = peer.config.get_peer(host_pubkey)

            # If either host or peer has a PresharedKey already, use it
            host_psk = host.config.get_peer(peer_pubkey).get("PresharedKey")
            peer_psk = peer.config.get_peer(host_pubkey).get("PresharedKey")
            debug("host_psk", host_psk)
            debug("peer_psk", peer_psk)
            # If they are different, but not None, raise an error
            if host_psk and peer_psk and host_psk != peer_psk:
                raise ValueError("PresharedKey mismatch")
            # If one of them is None, use the other
            if not host_psk and peer_psk:
                debug("setting host_psk")
                host.config.add_attr(peer_pubkey, "PresharedKey", peer_psk)
            if not peer_psk and host_psk:
                debug("setting peer_psk")
                peer.config.add_attr(host_pubkey, "PresharedKey", host_psk)
            # If both are None, generate a new one
            if not host_psk and not peer_psk:
                debug("generating psk")
                psk = wgexec.generate_presharedkey()
                host.config.add_attr(peer_pubkey, "PresharedKey", psk)
                peer.config.add_attr(host_pubkey, "PresharedKey", psk)
            # Now that we got the hard part out of the way,
            # let's ensure Endpoint, AllowedIPs and PersistentKeepalive are correct

            pairs_to_reconcile = {
                "Endpoint": f"{peer.ip}:{peer.port}",
                "AllowedIPs": f"{peer.wg_ip}/32",
                "PersistentKeepalive": 25,
            }
            for key, wanted_value in pairs_to_reconcile.items():
                peer_section = host.config.get_peer(peer_pubkey)
                debug(f"value for peer_section is {peer_section}")
                if (value := host.config.get_peer(peer_pubkey).get(key)) != wanted_value:
                    # if it is not None, delete it
                    debug(f"value for {key} is {value}")
                    if value:
                        debug(f"deleting {key} for {peer.name}")
                        host.config.del_attr(peer_pubkey, key)

                    debug(f"setting {key}")
                    host.config.add_attr(peer_pubkey, key, wanted_value)
            host._write()
            peer._write()


def main():
    _init()
    _reconcile_peers()


if __name__ == "__main__":
    main()
