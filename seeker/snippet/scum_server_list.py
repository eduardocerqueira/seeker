#date: 2023-04-03T16:54:33Z
#url: https://api.github.com/gists/50fc916904ccacf8a8a010e3d05aac65
#owner: https://api.github.com/users/iMoD1998

import struct
import socket
import ipaddress
import pprint

from typing import Optional

SERVER_INFO_STRUCT = struct.Struct("<I H 100s x B B B 9s I H B B")
SERVER_INFO_STRUCT_SIZE = SERVER_INFO_STRUCT.size

def parse_server_info(server_info_raw: bytes):
    server_ip, server_port, server_name, server_player_count, server_player_max_count, server_time_hour, server_unk, server_version_build, server_version_patch, server_version_minor, server_version_major = SERVER_INFO_STRUCT.unpack(server_info_raw)
    server_name = server_name.decode("latin-1").rstrip("\x00")
    
    return {
        "name": server_name,
        "ip": str(ipaddress.ip_address(server_ip)),
        "port": server_port,
        "player_count": server_player_count,
        "max_player_count": server_player_max_count,
        "time": server_time_hour,
        "version": f"{server_version_major}.{server_version_minor}.{server_version_patch}.{server_version_build}",
        "unk": server_unk
    }

def connect_to_master_server(ip: str, port: int) -> Optional[socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((ip, port))
        return sock
    except Exception as e:
        print(f"Failed to connect to {ip}:{port}: {e}")
        sock.close()
        return None

master_servers = [
    ("176.57.138.2", 1040),
    ("172.107.16.215", 1040),
    ("206.189.248.133", 1040)
]

def get_server_list() -> dict[str, dict]:
    #
    # Try to connect to the first master server we can reach.
    #
    sock = None
    for ip, port in master_servers:
        print(f"Trying to connect to {ip}:{port}")
        sock = connect_to_master_server(ip, port)
        if sock:
            break

    if not sock:
        print("Failed to connect to any of the master servers")
        exit(1)

    #
    # Send command to retrieve server list
    #
    sock.send(b"\x04\x03\x00\x00")

    #
    # First two bytes are the number of servers that are being sent.
    #
    num_servers = struct.unpack("<H", sock.recv(2))[0]

    #
    # Read the whole server list.
    #
    server_data_raw = b""
    while len(server_data_raw) < (num_servers * SERVER_INFO_STRUCT_SIZE):
        chunk = sock.recv(1024)
        if not chunk:
            break
        server_data_raw += chunk

    #
    # process each server info item
    #
    server_list = {}
    for i in range(0, len(server_data_raw), SERVER_INFO_STRUCT_SIZE):
        server_info_raw = server_data_raw[i:i+SERVER_INFO_STRUCT_SIZE]
        server_info = parse_server_info(server_info_raw)
        server_list[f"{server_info['ip']}:{server_info['port']}"] = server_info

    return server_list

server_list = get_server_list()

pprint.pprint(server_list["194.140.197.202:32002"])