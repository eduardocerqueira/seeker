#date: 2025-06-02T16:56:41Z
#url: https://api.github.com/gists/da873f327510fd546186c8b4fc6870c8
#owner: https://api.github.com/users/rc0j

# converts /etc/host -> ssh config 
import os

def parse_etc_hosts(file_path='hosts'):
    """ Parse /etc/hosts and return a list of (hostname, aliases) tuples. """
    hosts = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                parts = line.split()
                ip_address = parts[0]
                hostnames = parts[1:]
                for hostname in hostnames:
                    hosts.append((hostname, ip_address))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return hosts

def convert_to_ssh_config(hosts):
    """ Convert the parsed /etc/hosts entries into SSH config format. """
    ssh_config_lines = []
    for hostname, ip_address in hosts:
        ssh_config_lines.append(f"Host {hostname}")
        ssh_config_lines.append(f"  HostName {ip_address}")
        ssh_config_lines.append("")  # Blank line for separation
    return "\n".join(ssh_config_lines)

def write_ssh_config(config, file_path="config"):
    """ Write the SSH config string to the specified file. """
    file_path = os.path.expanduser(file_path)
    try:
        with open(file_path, 'w') as f:
            f.write(config)
        print(f"SSH config written to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def main():
    hosts = parse_etc_hosts()
    if hosts:
        ssh_config = convert_to_ssh_config(hosts)
        write_ssh_config(ssh_config)
    else:
        print("No hosts found to convert.")

if __name__ == "__main__":
    main()

