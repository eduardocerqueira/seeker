#date: 2024-11-26T17:10:28Z
#url: https://api.github.com/gists/09460fff1e1b02a0ded2c1f722bbc861
#owner: https://api.github.com/users/User087

import os
import json
import time
import http.client
import subprocess
import argparse

# Configuration
config_dir = "D:\prot"  # Directory where your config files are stored
wireguard_cmd = "wireguard /installtunnelservice"  # Command to connect to a WireGuard tunnel
wireguard_uninstall_cmd = "wireguard /uninstalltunnelservice"  # Command to disconnect a WireGuard tunnel

def fetch_logical_servers():
    """Fetch logical server information from the ProtonVPN public API."""
    connection = http.client.HTTPSConnection("account.proton.me")
    connection.request("GET", "/api/vpn/logicals")
    response = connection.getresponse()
    if response.status != 200:
        raise Exception(f"Failed to fetch logical servers: {response.status} {response.reason}")
    servers = json.loads(response.read().decode())["LogicalServers"]
    return servers

def get_existing_configs():
    """Get the list of existing configuration names without the file extension."""
    return [f.split(".")[0] for f in os.listdir(config_dir) if f.endswith(".conf")]

def extract_server_name(config_name):
    """Extract the relevant part of the configuration name to match it with the server names."""
    parts = config_name.split('-')
    server_number = parts[-1]  # Get the last part after the last hyphen
    return f"{parts[-3]}-{parts[-2]}#{server_number}"  # Construct the server name in the format US-IL#55

def find_best_servers(servers, existing_configs, location=None):
    """Find the best and second best servers based on load and existing configurations."""
    best_server = None
    second_best_server = None
    lowest_load = float('inf')
    second_lowest_load = float('inf')

    for server in servers:
        if location and not server['Name'].startswith(location):
            continue
        for config in existing_configs:
            config_server_name = extract_server_name(config)
            if server['Name'] == config_server_name:
                if server["Load"] < lowest_load:
                    second_best_server = best_server
                    second_lowest_load = lowest_load
                    best_server = server
                    lowest_load = server["Load"]
                elif server["Load"] < second_lowest_load:
                    second_best_server = server
                    second_lowest_load = server["Load"]
    
    return best_server, second_best_server

def check_current_connection():
    """Check if there is a current WireGuard connection."""
    command = "powershell -Command \"Get-NetAdapter | Where-Object {$_.InterfaceDescription -like '*WireGuard*'} | Select-Object -ExpandProperty Name\""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    adapters = result.stdout.splitlines()
    if adapters:
        tunnel_name = adapters[0]
        return tunnel_name, adapters
    return None, adapters

def disconnect_current_connection(tunnel_name):
    """Disconnect the current WireGuard connection."""
    command = f"{wireguard_uninstall_cmd} \"{tunnel_name}\""
    print(f"Disconnecting from {tunnel_name} with command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Disconnected from {tunnel_name} successfully.")
        # Wait for the adapter to be removed
        time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"Failed to disconnect: {e}")

def connect_to_server(server):
    """Connect to the server using WireGuard."""
    config_name = server['Name'].replace('#', '-')
    config_path = os.path.join(config_dir, f"Wire_AuTo-{config_name}.conf")
    command = f"{wireguard_cmd} \"{config_path}\""
    print(f"Connecting to {server['Name']} with load {server['Load']} using config {config_path}...")
    print(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print("Connected successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to connect: {e}")

def main(location):
    servers = fetch_logical_servers()
    existing_configs = get_existing_configs()

    # Create a dictionary to map server names to their loads
    server_loads = {server['Name']: server['Load'] for server in servers}

    print("\nExisting configurations:")
    for config in existing_configs:
        config_server_name = extract_server_name(config)
        load = server_loads.get(config_server_name, "Unknown")
        print(f"{config}: Load {load}")

    # Check current connection and print adapters
    current_tunnel, adapters = check_current_connection()
    print("\nNetwork adapters:")
    for adapter in adapters:
        print(adapter)

    print("\nChecking for matches:")
    best_server, second_best_server = find_best_servers(servers, existing_configs, location)

    if current_tunnel:
        current_server_name = extract_server_name(current_tunnel)
        print(f"Currently connected to: {current_server_name}")
        
        if current_server_name == best_server['Name']:
            print(f"Already connected to the best server: {best_server['Name']} with load {best_server['Load']}")
            if second_best_server:
                print(f"Connecting to the second best server: {second_best_server['Name']} with load {second_best_server['Load']}")
                disconnect_current_connection(current_tunnel)
                connect_to_server(second_best_server)
        else:
            print(f"Disconnecting from current server {current_server_name} and connecting to the best server {best_server['Name']} with load {best_server['Load']}")
            disconnect_current_connection(current_tunnel)
            connect_to_server(best_server)
    else:
        if best_server:
            connect_to_server(best_server)
        else:
            print("No suitable server found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect to the best ProtonVPN server based on load.")
    parser.add_argument('-location', type=str, help="Preferred server location (e.g., US-NY)")
    args = parser.parse_args()

    main(args.location)

