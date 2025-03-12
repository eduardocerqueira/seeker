#date: 2025-03-12T17:02:16Z
#url: https://api.github.com/gists/7d5db85f40b858494d1f759cc8f22a6d
#owner: https://api.github.com/users/ehzawad

#!/usr/bin/env python3
import socket
import requests
import concurrent.futures
import time
import sys
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import List, Tuple
import subprocess

@dataclass
class HostEntry:
    ip: str
    domains: List[str]
    
@dataclass
class TestResult:
    ip: str
    domain: str
    ip_reachable: bool
    domain_resolves: bool
    expected_ip_match: bool
    http_accessible: bool
    response_time: float = 0.0
    status_code: int = 0
    error: str = ""

def parse_hosts_entries(entries_text: str) -> List[HostEntry]:
    """Parse text of hosts file entries into structured data."""
    entries = []
    lines = entries_text.strip().split('\n')
    
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0].startswith('#'):
            continue
            
        ip = parts[0]
        domains = parts[1:]
        entries.append(HostEntry(ip=ip, domains=domains))
    
    return entries

def test_ip_reachable(ip: str) -> bool:
    """Test if an IP address is reachable."""
    try:
        # Use ping with a short timeout
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', ip],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False

def test_domain_resolution(domain: str) -> Tuple[bool, str]:
    """Test if a domain resolves to any IP."""
    try:
        ip = socket.gethostbyname(domain)
        return True, ip
    except socket.gaierror:
        return False, ""

def test_http_access(ip: str, domain: str) -> Tuple[bool, float, int, str]:
    """Test HTTP accessibility of a domain through specific IP."""
    # Create a properly formatted URL
    url = f"https://{domain}"
    
    try:
        # Set short timeouts to avoid hanging
        start_time = time.time()
        response = requests.get(
            url, 
            headers={"Host": domain},
            timeout=3,
            verify=False  # Disable SSL verification when testing
        )
        elapsed = time.time() - start_time
        
        return True, elapsed, response.status_code, ""
    except requests.RequestException as e:
        return False, 0.0, 0, str(e)

def test_entry(entry: HostEntry) -> List[TestResult]:
    """Test connectivity for all domains in a host entry."""
    results = []
    
    # First test if the IP is reachable
    ip_reachable = test_ip_reachable(entry.ip)
    
    for domain in entry.domains:
        # Test if the domain resolves
        domain_resolves, resolved_ip = test_domain_resolution(domain)
        expected_ip_match = (resolved_ip == entry.ip) if domain_resolves else False
        
        # Test HTTP accessibility
        http_accessible, response_time, status_code, error = test_http_access(entry.ip, domain)
        
        results.append(TestResult(
            ip=entry.ip,
            domain=domain,
            ip_reachable=ip_reachable,
            domain_resolves=domain_resolves,
            expected_ip_match=expected_ip_match,
            http_accessible=http_accessible,
            response_time=response_time,
            status_code=status_code,
            error=error
        ))
    
    return results

def print_results(all_results: List[TestResult]):
    """Print results in a readable format."""
    print(f"\n{'=' * 80}")
    print(f"{'DOMAIN':<30} {'IP':<15} {'PING':<5} {'DNS':<5} {'MATCH':<5} {'HTTP':<5} {'CODE':<5} {'TIME':<8}")
    print(f"{'-' * 80}")
    
    recommended_entries = []
    
    for result in all_results:
        ping_status = "✅" if result.ip_reachable else "❌"
        dns_status = "✅" if result.domain_resolves else "❌"
        match_status = "✅" if result.expected_ip_match else "❌"
        http_status = "✅" if result.http_accessible else "❌"
        time_str = f"{result.response_time:.2f}s" if result.http_accessible else "—"
        
        print(f"{result.domain:<30} {result.ip:<15} {ping_status:<5} {dns_status:<5} {match_status:<5} {http_status:<5} {result.status_code if result.status_code else '—':<5} {time_str:<8}")
        
        # Recommend this entry if either:
        # 1. IP is reachable and domain doesn't resolve correctly
        # 2. IP is reachable and HTTP is accessible
        if result.ip_reachable and (not result.domain_resolves or not result.expected_ip_match or result.http_accessible):
            recommended_entries.append(f"{result.ip} {result.domain}")
    
    print(f"\n{'=' * 80}")
    print("\nRECOMMENDED HOSTS ENTRIES:")
    print("\n".join(recommended_entries) if recommended_entries else "No entries recommended based on test results")

def main():
    # Disable SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Input hosts entries
    hosts_text = """151.101.65.140 reddit.com www.reddit.com
151.101.65.140 old.reddit.com
151.101.129.140 redd.it
151.101.1.140 i.reddit.com
151.101.193.140 i.redd.it
151.101.129.140 v.redd.it
151.101.129.140 preview.redd.it
151.101.193.140 styles.redditmedia.com
151.101.193.140 thumbs.redditmedia.com
151.101.65.140 external-preview.redd.it
151.101.193.140 gateway.reddit.com
151.101.65.140 oauth.reddit.com api.reddit.com"""
    
    entries = parse_hosts_entries(hosts_text)
    
    print(f"Testing {sum(len(entry.domains) for entry in entries)} domains across {len(entries)} IP addresses...")
    
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_entry = {executor.submit(test_entry, entry): entry for entry in entries}
        for future in concurrent.futures.as_completed(future_to_entry):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Error testing entry: {e}")
    
    # Sort results by domain for cleaner output
    all_results.sort(key=lambda r: r.domain)
    
    print_results(all_results)

if __name__ == "__main__":
    main()