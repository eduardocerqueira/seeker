#date: 2025-10-15T16:52:34Z
#url: https://api.github.com/gists/afdf5e4f384bb1674d4217f8a6ad1216
#owner: https://api.github.com/users/beye91

#!/usr/bin/env python3
import os, sys, csv, ipaddress, requests

API_BASE = os.getenv("TE_API_BASE", "https://api.thousandeyes.com").rstrip("/")
TOKEN = "**********"
HEADERS = {"Authorization": "**********": "application/json"}
AGENT_NAME = "thousandeyes-krakow-lab"

 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
    sys.exit("Missing THOUSANDEYES_TOKEN")

def get_agent_id():
    r = requests.get(f"{API_BASE}/v7/agents", headers=HEADERS, timeout=30)
    if r.status_code != 200:
        sys.exit(f"Failed to get agents: {r.status_code} {r.text}")
    for a in r.json().get("agents", []):
        if a.get("agentName", "").lower() == AGENT_NAME.lower():
            print(f"Using agent '{AGENT_NAME}' (ID {a['agentId']})")
            return int(a["agentId"])
    sys.exit(f"Agent '{AGENT_NAME}' not found")

def is_rfc1918(host):
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private
    except ValueError:
        return False  # hostname
       
def read_csv(path):
    pairs = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = [h.strip().lower() for h in next(reader)]
        targ_idx = next((i for i,h in enumerate(headers) if h in ("ip","target","address","host","hostname")), 0)
        name_idx = next((i for i,h in enumerate(headers) if h in ("name","title","label")), 1 if len(headers)>1 else 0)
        for row in reader:
            if not row: 
                continue
            target = row[targ_idx].strip() if len(row)>targ_idx else ""
            name = row[name_idx].strip() if len(row)>name_idx else target
            if target:
                pairs.append((target, name))
    return pairs

def create_icmp_test(server, name, agent_id):
    # Disable public BGP monitors for private targets
    disable_bgp = is_rfc1918(server)

    payload = {
        "testName": f"ICMP - {name}",
        "description": f"ICMP test to {server}",
        "interval": 300,
        "enabled": True,
        "protocol": "icmp",
        "server": server,
        "agents": [{"agentId": agent_id}],
        # Key bit: avoid using public BGP monitors with private (10/172/192.168) targets
        "bgpMeasurements": False if disable_bgp else True,
        "usePublicBgp": False if disable_bgp else True
    }

    r = requests.post(f"{API_BASE}/v7/tests/agent-to-server", headers=HEADERS, json=payload, timeout=30)
    if r.status_code >= 400:
        print(f"[ERR] {server} -> {r.status_code} {r.text}")
    else:
        data = r.json()
        tid = data.get("testId") or data.get("test", {}).get("testId")
        print(f"[OK] Created {payload['testName']} (ID {tid})")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python create_te_icmp_tests.py <csv_file>")
    agent_id = get_agent_id()
    items = read_csv(sys.argv[1])
    print(f"Found {len(items)} targets. Creating ICMP tests...\n")
    for server, name in items:
        create_icmp_test(server, name, agent_id)

if __name__ == "__main__":
    main()