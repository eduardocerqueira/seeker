#date: 2025-07-02T16:43:52Z
#url: https://api.github.com/gists/7e921773b1b6128de5d3006bbaaee2e0
#owner: https://api.github.com/users/Ashborn-o9

#!/usr/bin/env python3
import nmap
import requests
import json

# Optionally add your NVD API key here for increased rate limits
NVD_API_KEY = ""

def search_cves(cpe_name):
    headers = {}
    if NVD_API_KEY:
        headers['apiKey'] = NVD_API_KEY

    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cpeName={cpe_name}&resultsPerPage=3"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [
                item['cve']['id']
                for item in data.get('vulnerabilities', [])
            ]
        else:
            print(f"[!] NVD API error: {response.status_code}")
    except Exception as e:
        print(f"[!] Error fetching CVEs: {e}")
    return []

def scan_network(target_subnet):
    scanner = nmap.PortScanner()
    print(f"[*] Scanning network: {target_subnet}...")
    scanner.scan(hosts=target_subnet, arguments='-sV')

    report = {}
    for host in scanner.all_hosts():
        if scanner[host].state() == "up":
            print(f"\n[+] Host: {host}")
            report[host] = []
            for proto in scanner[host].all_protocols():
                lports = scanner[host][proto].keys()
                for port in lports:
                    srv = scanner[host][proto][port]
                    cpe = srv.get('cpe', '')
                    cves = search_cves(cpe) if cpe else []
                    print(f"  - Port {port}/{proto}: {srv['name']} ({srv.get('product', '')} {srv.get('version', '')})")
                    if cves:
                        print(f"    └ CVEs: {', '.join(cves)}")
                    else:
                        print("    └ No CVEs found or missing CPE")
                    report[host].append({
                        'port': port, 'protocol': proto, 'service': srv['name'],
                        'product': srv.get('product', ''), 'version': srv.get('version', ''),
                        'cpe': cpe, 'cves': cves
                    })
    return report

def main():
    target = input("Enter target subnet (e.g., 192.168.1.0/24): ").strip()
    result = scan_network(target)
    with open("vuln_scan_results.json", "w") as f:
        json.dump(result, f, indent=4)
    print("\n[+] Results saved to vuln_scan_results.json")

if __name__ == "__main__":
    main()
