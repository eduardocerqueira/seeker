#date: 2025-11-10T16:59:09Z
#url: https://api.github.com/gists/8fd808d9623b7ada91ffd1b68228036f
#owner: https://api.github.com/users/adujardin

#!/usr/bin/env python3
"""
Network Scanner - Scan your local network for devices
Displays IP addresses, MAC addresses, and checks for web ports (80/443)
Supports filtering by MAC address vendor prefix
Auto-detects CIDR from network interface configuration
Smart interface detection (skips Docker, loopback, virtual interfaces)
"""

import socket
import subprocess
import platform
import re
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import struct

def get_color_codes():
    """Returns color codes for terminal output"""
    return {
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'MAGENTA': '\033[95m',
        'BOLD': '\033[1m',
        'END': '\033[0m'
    }

def print_banner():
    """Print a nice banner"""
    colors = get_color_codes()
    print(f"\n{colors['CYAN']}{colors['BOLD']}{'='*60}")
    print("          🌐 Network Scanner Tool 🌐")
    print(f"{'='*60}{colors['END']}\n")

def normalize_mac(mac):
    """Normalize MAC address to uppercase without separators"""
    if not mac or mac == "N/A":
        return None
    # Remove all common separators (: - .) and convert to uppercase
    return re.sub(r'[:\-.]', '', mac).upper()

def mac_matches_prefix(mac, prefix):
    """
    Check if MAC address matches the given prefix
    Supports partial prefixes of any length
    """
    normalized_mac = normalize_mac(mac)
    normalized_prefix = normalize_mac(prefix)
    
    if not normalized_mac or not normalized_prefix:
        return False
    
    # Check if MAC starts with the prefix
    return normalized_mac.startswith(normalized_prefix)

def get_netmask_from_cidr(cidr):
    """Convert CIDR to netmask (e.g., 24 -> 255.255.255.0)"""
    mask = (0xffffffff >> (32 - cidr)) << (32 - cidr)
    return socket.inet_ntoa(struct.pack('>I', mask))

def get_cidr_from_netmask(netmask):
    """Convert netmask to CIDR (e.g., 255.255.255.0 -> 24)"""
    try:
        return sum([bin(int(x)).count('1') for x in netmask.split('.')])
    except:
        return None

def should_skip_interface(interface_name):
    """
    Determine if an interface should be skipped
    Skips: loopback, Docker, virtual interfaces, bridges, etc.
    """
    skip_patterns = [
        r'^lo$',           # loopback
        r'^docker\d*$',    # docker0, docker1, etc.
        r'^br-',           # docker bridge interfaces
        r'^veth',          # virtual ethernet (docker containers)
        r'^virbr',         # virtual bridge (libvirt/KVM)
        r'^vmnet',         # VMware interfaces
        r'^vboxnet',       # VirtualBox interfaces
        r'^tun',           # tunnel interfaces
        r'^tap',           # tap interfaces
    ]
    
    interface_lower = interface_name.lower()
    
    for pattern in skip_patterns:
        if re.match(pattern, interface_lower):
            return True
    
    return False

def get_interface_priority(interface_name):
    """
    Assign priority to interfaces
    Lower number = higher priority
    Ethernet > WiFi > Others
    """
    interface_lower = interface_name.lower()
    
    # Ethernet interfaces (highest priority)
    ethernet_patterns = [r'^eth\d*$', r'^en[ops]\d*', r'^eno\d+$', r'^enp\d+s\d+$', r'^ens\d+$']
    for pattern in ethernet_patterns:
        if re.match(pattern, interface_lower):
            return 1
    
    # WiFi interfaces (second priority)
    wifi_patterns = [r'^wlan\d*$', r'^wlp\d+s\d+$', r'^wl[ops]\d*', r'^wifi\d*$']
    for pattern in wifi_patterns:
        if re.match(pattern, interface_lower):
            return 2
    
    # Everything else
    return 3

def get_all_interfaces_linux():
    """Get all network interfaces on Linux with their IP and CIDR"""
    interfaces = []
    
    try:
        # Try 'ip addr' first (modern Linux)
        output = subprocess.check_output("ip addr", shell=True).decode()
        
        current_interface = None
        for line in output.split('\n'):
            # Interface line: "2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> ..."
            if re.match(r'^\d+:', line):
                match = re.search(r'^\d+:\s+(\S+):', line)
                if match:
                    current_interface = match.group(1)
            
            # IP address line: "    inet 192.168.1.100/24 ..."
            elif current_interface and 'inet ' in line and '/' in line:
                match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)/(\d+)', line)
                if match:
                    ip_address = match.group(1)
                    cidr = int(match.group(2))
                    
                    # Skip loopback and link-local
                    if not ip_address.startswith(('127.', '169.254.')):
                        interfaces.append({
                            'name': current_interface,
                            'ip': ip_address,
                            'cidr': cidr,
                            'skip': should_skip_interface(current_interface),
                            'priority': get_interface_priority(current_interface)
                        })
    except:
        # Fallback to ifconfig
        try:
            output = subprocess.check_output("ifconfig", shell=True).decode()
            
            current_interface = None
            for line in output.split('\n'):
                # Interface line starts at column 0
                if line and not line[0].isspace():
                    match = re.match(r'^(\S+)', line)
                    if match:
                        current_interface = match.group(1).rstrip(':')
                
                # IP line is indented
                elif current_interface and 'inet ' in line:
                    ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
                    mask_match = re.search(r'netmask (\d+\.\d+\.\d+\.\d+)', line)
                    
                    if ip_match and mask_match:
                        ip_address = ip_match.group(1)
                        netmask = mask_match.group(1)
                        cidr = get_cidr_from_netmask(netmask)
                        
                        if not ip_address.startswith(('127.', '169.254.')):
                            interfaces.append({
                                'name': current_interface,
                                'ip': ip_address,
                                'cidr': cidr,
                                'skip': should_skip_interface(current_interface),
                                'priority': get_interface_priority(current_interface)
                            })
        except:
            pass
    
    return interfaces

def get_all_interfaces_mac():
    """Get all network interfaces on macOS with their IP and CIDR"""
    interfaces = []
    
    try:
        output = subprocess.check_output("ifconfig", shell=True).decode()
        
        current_interface = None
        for line in output.split('\n'):
            # Interface line starts at column 0
            if line and not line[0].isspace():
                match = re.match(r'^(\S+):', line)
                if match:
                    current_interface = match.group(1)
            
            # IP line is indented
            elif current_interface and '\tinet ' in line:
                match = re.search(r'inet (\d+\.\d+\.\d+\.\d+).*netmask (0x[0-9a-fA-F]+)', line)
                if match:
                    ip_address = match.group(1)
                    netmask_hex = match.group(2)
                    
                    if not ip_address.startswith(('127.', '169.254.')):
                        # Convert hex netmask to CIDR
                        mask_int = int(netmask_hex, 16)
                        cidr = bin(mask_int).count('1')
                        
                        interfaces.append({
                            'name': current_interface,
                            'ip': ip_address,
                            'cidr': cidr,
                            'skip': should_skip_interface(current_interface),
                            'priority': get_interface_priority(current_interface)
                        })
    except:
        pass
    
    return interfaces

def get_all_interfaces_windows():
    """Get all network interfaces on Windows with their IP and CIDR"""
    interfaces = []
    
    try:
        output = subprocess.check_output("ipconfig /all", shell=True).decode(errors='ignore')
        
        lines = output.split('\n')
        current_adapter = None
        current_ip = None
        current_mask = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Adapter name line
            if 'adapter' in line.lower() and ':' in line:
                # Save previous adapter if we have one
                if current_adapter and current_ip and current_mask:
                    if not current_ip.startswith(('127.', '169.254.')):
                        cidr = get_cidr_from_netmask(current_mask)
                        interfaces.append({
                            'name': current_adapter,
                            'ip': current_ip,
                            'cidr': cidr,
                            'skip': should_skip_interface(current_adapter),
                            'priority': get_interface_priority(current_adapter)
                        })
                
                # Extract adapter name
                match = re.search(r'adapter (.+?):', line)
                if match:
                    current_adapter = match.group(1).strip()
                current_ip = None
                current_mask = None
            
            # IPv4 address
            elif 'IPv4' in line_stripped and '.' in line_stripped:
                match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line_stripped)
                if match:
                    current_ip = match.group(1)
            
            # Subnet mask
            elif 'Subnet Mask' in line_stripped:
                match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line_stripped)
                if match:
                    current_mask = match.group(1)
        
        # Don't forget the last adapter
        if current_adapter and current_ip and current_mask:
            if not current_ip.startswith(('127.', '169.254.')):
                cidr = get_cidr_from_netmask(current_mask)
                interfaces.append({
                    'name': current_adapter,
                    'ip': current_ip,
                    'cidr': cidr,
                    'skip': should_skip_interface(current_adapter),
                    'priority': get_interface_priority(current_adapter)
                })
    except:
        pass
    
    return interfaces

def get_all_interfaces():
    """Get all network interfaces based on OS"""
    os_name = platform.system().lower()
    
    if os_name == 'windows':
        return get_all_interfaces_windows()
    elif os_name == 'darwin':  # macOS
        return get_all_interfaces_mac()
    else:  # Linux
        return get_all_interfaces_linux()

def select_best_interface(interfaces, specified_interface=None):
    """
    Select the best interface from the list
    Priority: specified > ethernet > wifi > others
    Skips Docker, loopback, and virtual interfaces by default
    """
    colors = get_color_codes()
    
    if not interfaces:
        return None
    
    # If user specified an interface, try to find it
    if specified_interface:
        for iface in interfaces:
            if iface['name'] == specified_interface:
                return iface
        print(f"{colors['RED']}Error: Interface '{specified_interface}' not found{colors['END']}")
        return None
    
    # Filter out interfaces we should skip
    good_interfaces = [iface for iface in interfaces if not iface['skip']]
    
    if not good_interfaces:
        print(f"{colors['YELLOW']}Warning: Only virtual/docker interfaces found, using them anyway{colors['END']}")
        good_interfaces = interfaces
    
    # Sort by priority (ethernet first, then wifi, then others)
    good_interfaces.sort(key=lambda x: x['priority'])
    
    return good_interfaces[0]

def list_interfaces():
    """List all available network interfaces"""
    colors = get_color_codes()
    interfaces = get_all_interfaces()
    
    if not interfaces:
        print(f"{colors['RED']}No network interfaces found{colors['END']}")
        return
    
    print(f"\n{colors['CYAN']}{colors['BOLD']}Available Network Interfaces:{colors['END']}\n")
    print(f"{colors['BOLD']}{'Interface':<20} {'IP Address':<16} {'CIDR':<6} {'Type':<15} {'Status'}{colors['END']}")
    print(f"{'-'*80}")
    
    for iface in interfaces:
        # Determine type
        if iface['priority'] == 1:
            iface_type = f"{colors['GREEN']}Ethernet{colors['END']}"
        elif iface['priority'] == 2:
            iface_type = f"{colors['BLUE']}WiFi{colors['END']}"
        else:
            iface_type = "Other"
        
        # Determine if it would be skipped
        if iface['skip']:
            status = f"{colors['YELLOW']}(skipped){colors['END']}"
        else:
            status = f"{colors['GREEN']}(usable){colors['END']}"
        
        print(f"{iface['name']:<20} {iface['ip']:<16} /{iface['cidr']:<5} {iface_type:<24} {status}")
    
    print()

def get_network_info(specified_interface=None):
    """
    Get network configuration (IP and CIDR) for the best interface
    Returns: (interface_name, ip_address, cidr)
    """
    colors = get_color_codes()
    
    interfaces = get_all_interfaces()
    
    if not interfaces:
        print(f"{colors['YELLOW']}Warning: Could not detect network interfaces{colors['END']}")
        return None, get_local_ip(), 24
    
    selected = select_best_interface(interfaces, specified_interface)
    
    if not selected:
        return None, get_local_ip(), 24
    
    return selected['name'], selected['ip'], selected['cidr']

def get_local_ip():
    """Get the local IP address of this machine (fallback method)"""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None

def calculate_network_range(ip_address, cidr):
    """Calculate network range from IP address and CIDR"""
    try:
        network = ipaddress.IPv4Network(f"{ip_address}/{cidr}", strict=False)
        return str(network)
    except Exception as e:
        print(f"Error calculating network range: {e}")
        return None

def get_mac_address(ip):
    """Get MAC address for a given IP"""
    try:
        os_name = platform.system().lower()
        
        if os_name == "windows":
            # Windows: use arp -a
            output = subprocess.check_output(f"arp -a {ip}", shell=True).decode()
            mac_search = re.search(r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})", output)
        else:
            # Linux/Mac: use arp -n or ip neigh
            try:
                output = subprocess.check_output(f"arp -n {ip}", shell=True).decode()
            except:
                output = subprocess.check_output(f"ip neigh show {ip}", shell=True).decode()
            mac_search = re.search(r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})", output)
        
        if mac_search:
            return mac_search.group(0).upper()
    except:
        pass
    return "N/A"

def check_port(ip, port, timeout=0.5):
    """Check if a specific port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def ping_host(ip):
    """Check if host is alive using ping"""
    try:
        os_name = platform.system().lower()
        
        if os_name == "windows":
            command = f"ping -n 1 -w 500 {ip}"
        else:
            command = f"ping -c 1 -W 1 {ip}"
        
        output = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2
        )
        return output.returncode == 0
    except:
        return False

def scan_host(ip, mac_filter=None):
    """Scan a single host for information"""
    if not ping_host(ip):
        return None
    
    mac = get_mac_address(ip)
    
    # Apply MAC filter if specified
    if mac_filter and not mac_matches_prefix(mac, mac_filter):
        return None
    
    port_80 = check_port(ip, 80)
    port_443 = check_port(ip, 443)
    
    return {
        'ip': ip,
        'mac': mac,
        'port_80': port_80,
        'port_443': port_443
    }

def scan_network(network_range, mac_filter=None, max_workers=50):
    """Scan all hosts in the network range"""
    colors = get_color_codes()
    
    try:
        network = ipaddress.IPv4Network(network_range, strict=False)
        hosts = list(network.hosts())
        
        print(f"{colors['YELLOW']}Scanning {len(hosts)} hosts in {network_range}...")
        if mac_filter:
            print(f"Filtering by MAC prefix: {colors['BOLD']}{mac_filter}{colors['END']}{colors['YELLOW']}")
        print(f"This may take a few minutes...{colors['END']}\n")
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ip = {executor.submit(scan_host, str(ip), mac_filter): str(ip) for ip in hosts}
            
            for future in as_completed(future_to_ip):
                completed += 1
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{len(hosts)} hosts scanned...", end='\r')
                
                result = future.result()
                if result:
                    results.append(result)
        
        print(f"\n{colors['GREEN']}Scan complete!{colors['END']}\n")
        return results
        
    except Exception as e:
        print(f"{colors['RED']}Error during scan: {e}{colors['END']}")
        return []

def display_results(results, mac_filter=None):
    """Display scan results in a user-friendly format"""
    colors = get_color_codes()
    
    if not results:
        if mac_filter:
            print(f"{colors['RED']}No devices found matching MAC prefix: {mac_filter}{colors['END']}")
        else:
            print(f"{colors['RED']}No devices found on the network.{colors['END']}")
        return
    
    print(f"{colors['BOLD']}{colors['CYAN']}{'='*80}")
    print(f"{'IP Address':<18} {'MAC Address':<20} {'Port 80':<12} {'Port 443':<12}")
    print(f"{'='*80}{colors['END']}")
    
    for device in sorted(results, key=lambda x: ipaddress.IPv4Address(x['ip'])):
        ip = device['ip']
        mac = device['mac']
        port_80 = f"{colors['GREEN']}✓ OPEN{colors['END']}" if device['port_80'] else f"{colors['RED']}✗ Closed{colors['END']}"
        port_443 = f"{colors['GREEN']}✓ OPEN{colors['END']}" if device['port_443'] else f"{colors['RED']}✗ Closed{colors['END']}"
        
        # Highlight matching MAC prefix if filter is applied
        if mac_filter:
            normalized_prefix = normalize_mac(mac_filter)
            if normalized_prefix and mac != "N/A":
                normalized_mac = normalize_mac(mac)
                # Format the MAC with highlighting
                prefix_len = len(normalized_prefix)
                highlighted_mac = f"{colors['YELLOW']}{colors['BOLD']}{mac[:prefix_len*2 + prefix_len//2]}{colors['END']}{mac[prefix_len*2 + prefix_len//2:]}"
                mac = highlighted_mac
        
        # Add extra spacing for ANSI color codes
        print(f"{ip:<18} {mac:<35} {port_80:<20} {port_443:<20}")
    
    print(f"\n{colors['BOLD']}Total devices found: {len(results)}{colors['END']}")
    
    # Summary of web servers
    web_servers = [d for d in results if d['port_80'] or d['port_443']]
    if web_servers:
        print(f"\n{colors['GREEN']}Devices with web services (80/443):{colors['END']}")
        for device in web_servers:
            protocols = []
            if device['port_80']:
                protocols.append(f"http://{device['ip']}")
            if device['port_443']:
                protocols.append(f"https://{device['ip']}")
            print(f"  • {device['ip']} ({device['mac']}) - {', '.join(protocols)}")

def print_common_vendors():
    """Print some common MAC vendor prefixes"""
    colors = get_color_codes()
    print(f"\n{colors['CYAN']}Common MAC Vendor Prefixes:{colors['END']}")
    vendors = [
        ("Apple", "00:03:93, 00:05:02, 00:0A:27, 00:0D:93, etc."),
        ("Dell", "00:14:22, 00:1E:C9, 18:03:73, etc."),
        ("HP", "00:1F:29, 00:21:5A, 00:23:7D, etc."),
        ("Cisco", "00:0A:B7, 00:0F:23, 00:1B:D4, etc."),
        ("Intel", "00:13:20, 00:15:00, 00:1B:21, etc."),
        ("Asus", "00:1F:C6, 00:22:15, 00:26:18, etc."),
        ("Samsung", "00:12:47, 00:15:B9, 00:1A:8A, etc."),
        ("TP-Link", "00:27:19, 50:C7:BF, F4:F2:6D, etc."),
        ("Raspberry Pi", "B8:27:EB, DC:A6:32, E4:5F:01"),
    ]
    for vendor, prefixes in vendors:
        print(f"  {colors['YELLOW']}{vendor:<15}{colors['END']} {prefixes}")
    print(f"\n{colors['BLUE']}Tip: You can use any partial prefix (e.g., '00', '00:03', 'B827', etc.){colors['END']}\n")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Network Scanner - Scan local network for devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Auto-detect network and scan
  %(prog)s --interface eth0                   # Use specific interface
  %(prog)s --list-interfaces                  # Show all interfaces
  %(prog)s --mac-filter B8:27:EB              # Scan only Raspberry Pi devices
  %(prog)s --mac-filter B8                    # Scan devices with MAC starting with B8
  %(prog)s --network 192.168.1.0/24           # Specify network range manually
  %(prog)s --interface wlan0 --mac-filter 00:03:93  # Combine options
  %(prog)s --list-vendors                     # Show common vendor prefixes
  %(prog)s --show-config                      # Show detected network configuration
        '''
    )
    
    parser.add_argument(
        '--interface', '-i',
        type=str,
        help='Specify network interface to use (e.g., eth0, wlan0, enp3s0)'
    )
    
    parser.add_argument(
        '--list-interfaces',
        action='store_true',
        help='List all available network interfaces and exit'
    )
    
    parser.add_argument(
        '--mac-filter', '-m',
        type=str,
        help='Filter devices by MAC address prefix (supports partial match, e.g., "B8:27:EB", "B8", "00:03")'
    )
    
    parser.add_argument(
        '--network', '-n',
        type=str,
        help='Network range in CIDR notation (e.g., 192.168.1.0/24, 10.0.0.0/16)'
    )
    
    parser.add_argument(
        '--list-vendors', '-l',
        action='store_true',
        help='Show common MAC vendor prefixes and exit'
    )
    
    parser.add_argument(
        '--show-config', '-c',
        action='store_true',
        help='Show detected network configuration and exit'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=50,
        help='Number of concurrent scanning threads (default: 50)'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    colors = get_color_codes()
    args = parse_arguments()
    
    print_banner()
    
    # If user wants to list interfaces
    if args.list_interfaces:
        list_interfaces()
        sys.exit(0)
    
    # If user wants to see vendor list
    if args.list_vendors:
        print_common_vendors()
        sys.exit(0)
    
    # Get network configuration
    interface_name, local_ip, detected_cidr = get_network_info(args.interface)
    
    if not local_ip:
        print(f"{colors['RED']}Could not determine local IP address.{colors['END']}")
        sys.exit(1)
    
    netmask = get_netmask_from_cidr(detected_cidr) if detected_cidr else "Unknown"
    
    print(f"{colors['BLUE']}Detected Network Configuration:{colors['END']}")
    if interface_name:
        # Determine interface type color
        priority = get_interface_priority(interface_name)
        if priority == 1:
            iface_color = colors['GREEN']
            iface_type = "Ethernet"
        elif priority == 2:
            iface_color = colors['BLUE']
            iface_type = "WiFi"
        else:
            iface_color = colors['YELLOW']
            iface_type = "Other"
        
        print(f"  Interface:  {iface_color}{colors['BOLD']}{interface_name}{colors['END']} ({iface_type})")
    print(f"  IP Address: {colors['BOLD']}{local_ip}{colors['END']}")
    print(f"  Netmask:    {colors['BOLD']}{netmask}{colors['END']} (/{detected_cidr})")
    
    # Calculate network range
    if detected_cidr:
        auto_network_range = calculate_network_range(local_ip, detected_cidr)
        print(f"  Network:    {colors['BOLD']}{auto_network_range}{colors['END']}")
    
    # If user just wants to see config
    if args.show_config:
        print(f"\n{colors['CYAN']}Use --list-interfaces to see all available interfaces{colors['END']}")
        sys.exit(0)
    
    # Determine network range to scan
    if args.network:
        network_range = args.network
        print(f"\n{colors['YELLOW']}Using manually specified network: {colors['BOLD']}{network_range}{colors['END']}")
    else:
        if detected_cidr:
            network_range = auto_network_range
            print(f"\n{colors['GREEN']}Using auto-detected network range{colors['END']}")
        else:
            print(f"\n{colors['YELLOW']}Could not auto-detect CIDR, using /24 as default{colors['END']}")
            network_range = calculate_network_range(local_ip, 24)
    
    if not network_range:
        print(f"{colors['RED']}Invalid network range.{colors['END']}")
        sys.exit(1)
    
    # Display MAC filter if specified
    mac_filter = args.mac_filter
    if mac_filter:
        print(f"\n{colors['GREEN']}MAC Filter Active: {colors['BOLD']}{mac_filter}{colors['END']}")
        print(f"{colors['YELLOW']}Only showing devices with MAC addresses starting with this prefix{colors['END']}")
    
    # Perform scan
    results = scan_network(network_range, mac_filter, args.workers)
    
    # Display results
    display_results(results, mac_filter)
    
    print(f"\n{colors['CYAN']}{'='*80}{colors['END']}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        colors = get_color_codes()
        print(f"\n\n{colors['YELLOW']}Scan interrupted by user.{colors['END']}")
        sys.exit(0)