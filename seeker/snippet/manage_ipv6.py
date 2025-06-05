#date: 2025-06-05T17:06:09Z
#url: https://api.github.com/gists/a84540872002f80c1bb1d3621a81f961
#owner: https://api.github.com/users/cspence001

#!/usr/bin/env python3

import subprocess

# Define ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

def get_network_services():
    """Fetch the list of all network services."""
    try:
        result = subprocess.run(
            ["networksetup", "-listallnetworkservices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        # Skip the header and return the services
        return result.stdout.splitlines()[1:]
    except subprocess.CalledProcessError as e:
        print("Error fetching network services:", e)
        return []

def check_ipv6_status(service):
    """Check the IPv6 status for the specified network service."""
    try:
        result = subprocess.run(
            ["networksetup", "-getinfo", service],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        # Check for "IPv6: Off" in the output
        if "IPv6: Off" in result.stdout:
            return "disabled"
        elif "IPv6: Automatic" in result.stdout or "IPv6: Manually" in result.stdout:
            return "enabled"
        else:
            return "unknown"
    except subprocess.CalledProcessError as e:
        print(f"Failed to get IPv6 status for '{service}': {e}")
        return "unknown"

def enable_ipv6(service):
    """Enable IPv6 for the specified network service."""
    try:
        subprocess.run(
            ["sudo", "networksetup", "-setv6automatic", service],
            check=True,
        )
        print(f"Enabled IPv6 for '{service}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to enable IPv6 for '{service}': {e}")

def disable_ipv6(service):
    """Disable IPv6 for the specified network service."""
    try:
        subprocess.run(
            ["sudo", "networksetup", "-setv6off", service],
            check=True,
        )
        print(f"Disabled IPv6 for '{service}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to disable IPv6 for '{service}': {e}")

def main():
    network_services = get_network_services()

    for service in network_services:
        service = service.strip()  # Trim whitespace

        # Skip disabled services (those starting with '*')
        if service.startswith('*'):
            print(f"{service} is disabled, skipping.")
            continue

        ipv6_status = check_ipv6_status(service)

        if ipv6_status == "unknown":
            print(f"Unable to determine IPv6 status for {YELLOW}'{service}'{RESET} , skipping.")
            continue

        if ipv6_status == "enabled":
            # If IPv6 is enabled, prompt the user for action
            while True:
                choice = input(f"IPv6 is currently {GREEN}enabled{RESET} for {YELLOW}'{service}'{RESET}. Do you want to {BOLD}disable{RESET} IPv6 for '{service}'? (y/n): ").strip().lower()
                
                if choice in ['y', 'yes']:
                    disable_ipv6(service)
                    break
                elif choice in ['n', 'no']:
                    print(f"{service} will remain enabled.")
                    break
                else:
                    print("Invalid response. Please enter 'y' or 'n'.")

        elif ipv6_status == "disabled":
            # If IPv6 is disabled, prompt the user for action
            while True:
                choice = input(f"IPv6 is currently {RED}disabled{RESET} for {YELLOW}'{service}'{RESET}. Do you want to {BOLD}enable{RESET} it? (y/n): ").strip().lower()
                
                if choice in ['y', 'yes']:
                    enable_ipv6(service)
                    break
                elif choice in ['n', 'no']:
                    print(f"{service} will remain disabled.")
                    break
                else:
                    print("Invalid response. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()
