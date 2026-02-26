#date: 2026-02-26T17:35:55Z
#url: https://api.github.com/gists/2384b6cb742f6b6e09a710e7f2032e05
#owner: https://api.github.com/users/saeedseyfi

#!/usr/bin/env python3
"""
Domain availability checker using RDAP + WHOIS fallback.
Free, no API key needed.

Strategy:
  1. Try IANA bootstrap to find the RDAP server for the TLD
  2. If not in bootstrap, try rdap.org (universal redirect service)
  3. If no RDAP at all, fall back to WHOIS (port 43)

Usage:
    python check_domain.py evermynd.com
    python check_domain.py example.com example.se atahealth.co coolapp.dev
    python check_domain.py --list-tlds          # Show all RDAP-supported TLDs
"""

import sys
import os
import json
import time
import socket
import urllib.request
import urllib.error
from typing import Optional

BOOTSTRAP_URL = "https://data.iana.org/rdap/dns.json"
RDAP_ORG_BASE = "https://rdap.org"
CACHE_DIR = os.path.expanduser("~/.cache/rdap")
CACHE_FILE = os.path.join(CACHE_DIR, "bootstrap.json")
CACHE_MAX_AGE = 86400  # 24 hours

# Well-known WHOIS servers for TLDs without RDAP support.
# Used as last-resort fallback.
WHOIS_SERVERS: dict[str, str] = {
    "com": "whois.verisign-grs.com",
    "net": "whois.verisign-grs.com",
    "org": "whois.publicinterestregistry.org",
    "co": "whois.registry.co",
    "io": "whois.nic.io",
    "se": "whois.iis.se",
    "de": "whois.denic.de",
    "eu": "whois.eu",
    "it": "whois.nic.it",
    "ch": "whois.nic.ch",
    "be": "whois.dns.be",
    "nl": "whois.domain-registry.nl",
    "at": "whois.nic.at",
    "es": "whois.nic.es",
    "jp": "whois.jprs.jp",
    "us": "whois.nic.us",
    "me": "whois.nic.me",
    "uk": "whois.nic.uk",
    "co.uk": "whois.nic.uk",
    "ru": "whois.tcinet.ru",
    "br": "whois.registro.br",
    "in": "whois.nixiregistry.in",
    "nz": "whois.irs.net.nz",
    "cc": "ccwhois.verisign-grs.com",
    "tv": "whois.nic.tv",
    "ai": "whois.nic.ai",
}


# ── Bootstrap / TLD Map ─────────────────────────────────────────────

def fetch_bootstrap() -> dict:
    """Download the IANA RDAP bootstrap file."""
    req = urllib.request.Request(
        BOOTSTRAP_URL,
        headers={"Accept": "application/json", "User-Agent": "domain-checker/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def load_bootstrap() -> dict:
    """Load bootstrap with local file cache (refreshes every 24h)."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < CACHE_MAX_AGE:
            with open(CACHE_FILE) as f:
                return json.load(f)

    print("⏳ Fetching RDAP bootstrap from IANA...", file=sys.stderr)
    data = fetch_bootstrap()

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)

    return data


def build_tld_map(bootstrap: dict) -> dict[str, list[str]]:
    """Build TLD -> RDAP server URL mapping from IANA bootstrap."""
    tld_map: dict[str, list[str]] = {}
    for entry in bootstrap.get("services", []):
        tlds, urls = entry[0], entry[1]
        for tld in tlds:
            tld_map[tld.lower()] = urls
    return tld_map


def find_rdap_server(domain: str, tld_map: dict) -> Optional[str]:
    """Find the RDAP server for a domain using the IANA bootstrap."""
    parts = domain.lower().strip(".").split(".")
    if len(parts) < 2:
        return None

    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i + 1:])
        if candidate in tld_map:
            return tld_map[candidate][0]

    return None


def find_whois_server(domain: str) -> Optional[str]:
    """Find a WHOIS server for the domain's TLD."""
    parts = domain.lower().strip(".").split(".")
    if len(parts) < 2:
        return None

    # Try multi-level TLD first (co.uk)
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i + 1:])
        if candidate in WHOIS_SERVERS:
            return WHOIS_SERVERS[candidate]

    return None


# ── RDAP Check ───────────────────────────────────────────────────────

def check_rdap(domain: str, rdap_server: str) -> dict:
    """Query an RDAP server for a domain."""
    base = rdap_server.rstrip("/")
    url = f"{base}/domain/{domain}"

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/rdap+json, application/json",
            "User-Agent": "domain-checker/1.0",
        },
    )

    result = {
        "domain": domain,
        "available": None,
        "method": "rdap",
        "server": rdap_server,
    }

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            result["available"] = False
            result["status"] = data.get("status", [])
            result["events"] = [
                {"action": e.get("eventAction"), "date": e.get("eventDate")}
                for e in data.get("events", [])
                if e.get("eventAction") in ("registration", "expiration", "last changed")
            ]
            ns = data.get("nameservers", [])
            if ns:
                result["nameservers"] = [
                    n.get("ldhName", "") for n in ns if n.get("ldhName")
                ]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            result["available"] = True
        elif e.code == 429:
            result["error"] = "Rate limited — try again later"
        else:
            result["error"] = f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        result["error"] = f"Connection failed: {e.reason}"
    except Exception as e:
        result["error"] = f"Error: {e}"

    return result


def check_rdap_org(domain: str) -> dict:
    """
    Try rdap.org as a universal redirect/bootstrap service.
    It returns 302 to the correct RDAP server, or 404 if TLD unsupported.
    """
    url = f"{RDAP_ORG_BASE}/domain/{domain}"

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/rdap+json, application/json",
            "User-Agent": "domain-checker/1.0",
        },
    )

    result = {
        "domain": domain,
        "available": None,
        "method": "rdap.org",
        "server": RDAP_ORG_BASE,
    }

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            final_url = resp.url if hasattr(resp, "url") else url
            result["server"] = final_url
            result["available"] = False
            result["status"] = data.get("status", [])
            result["events"] = [
                {"action": e.get("eventAction"), "date": e.get("eventDate")}
                for e in data.get("events", [])
                if e.get("eventAction") in ("registration", "expiration", "last changed")
            ]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # 404 from rdap.org = TLD not supported by ANY RDAP server
            result["error"] = "TLD not supported by any RDAP server"
        elif e.code == 429:
            result["error"] = "rdap.org rate limited (max 10 req/10 sec)"
        else:
            result["error"] = f"HTTP {e.code}"
    except Exception as e:
        result["error"] = f"Error: {e}"

    return result


# ── WHOIS Fallback ───────────────────────────────────────────────────

def check_whois(domain: str, whois_server: str) -> dict:
    """
    Simple WHOIS lookup over TCP port 43.
    Parses the response to determine if the domain exists.
    """
    result = {
        "domain": domain,
        "available": None,
        "method": "whois",
        "server": whois_server,
    }

    try:
        sock = socket.create_connection((whois_server, 43), timeout=10)
        sock.sendall(f"{domain}\r\n".encode())

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        sock.close()

        text = response.decode("utf-8", errors="replace").lower()

        # Common "not found" patterns across different WHOIS servers
        not_found_patterns = [
            "no match",
            "not found",
            "no entries found",
            "no data found",
            "nothing found",
            "status: free",
            "status: available",
            "domain not found",
            "no information available",
            "is free",
            "no object found",
            "the queried object does not exist",
        ]

        if any(pattern in text for pattern in not_found_patterns):
            result["available"] = True
        else:
            result["available"] = False
            # Try to extract some useful info
            for line in response.decode("utf-8", errors="replace").splitlines():
                line_stripped = line.strip()
                lower = line_stripped.lower()
                if any(k in lower for k in ["creation date", "registered on", "created"]):
                    result.setdefault("events", []).append(
                        {"action": "registration", "date": line_stripped.split(":", 1)[-1].strip()}
                    )
                elif any(k in lower for k in ["expir", "renewal date", "paid-till"]):
                    result.setdefault("events", []).append(
                        {"action": "expiration", "date": line_stripped.split(":", 1)[-1].strip()}
                    )
                elif lower.startswith("nserver") or lower.startswith("name server"):
                    ns = line_stripped.split(":", 1)[-1].strip().split()[0]
                    result.setdefault("nameservers", []).append(ns)

    except socket.timeout:
        result["error"] = "WHOIS timeout"
    except Exception as e:
        result["error"] = f"WHOIS error: {e}"

    return result


# ── Main Check (tries strategies in order) ───────────────────────────

def check_domain(domain: str, tld_map: dict) -> dict:
    """
    Check domain availability using the best available method:
      1. Direct RDAP via IANA bootstrap
      2. rdap.org universal redirect
      3. WHOIS fallback (port 43)
    """
    domain = domain.lower().strip().strip(".")

    # Strategy 1: IANA bootstrap → direct RDAP
    rdap_server = find_rdap_server(domain, tld_map)
    if rdap_server:
        result = check_rdap(domain, rdap_server)
        if "error" not in result:
            return result
        # If direct RDAP errored, try rdap.org next

    # Strategy 2: rdap.org (knows more servers than IANA bootstrap)
    result = check_rdap_org(domain)
    if "error" not in result:
        return result

    # Strategy 3: WHOIS fallback
    whois_server = find_whois_server(domain)
    if whois_server:
        return check_whois(domain, whois_server)

    # Nothing worked
    tld = domain.split(".")[-1]
    return {
        "domain": domain,
        "available": None,
        "method": "none",
        "error": f"No RDAP or WHOIS server found for .{tld}",
    }


# ── Output ───────────────────────────────────────────────────────────

def print_result(result: dict) -> None:
    domain = result["domain"]
    method = result.get("method", "?")

    if "error" in result:
        print(f"  {domain}")
        print(f"    ⚠️  {result['error']}")
        if result.get("server"):
            print(f"    Method: {method} ({result['server']})")
        print()
        return

    if result["available"] is True:
        print(f"  {domain}")
        print(f"    ✅ AVAILABLE")
    elif result["available"] is False:
        print(f"  {domain}")
        print(f"    ❌ TAKEN")
        for s in result.get("status", []):
            print(f"       status: {s}")
        for ev in result.get("events", []):
            print(f"       {ev['action']}: {ev['date']}")
        for ns in result.get("nameservers", []):
            print(f"       ns: {ns}")
    else:
        print(f"  {domain}")
        print(f"    ❓ UNKNOWN (could not determine)")

    print(f"    Method: {method} ({result.get('server', '?')})")
    print()


def print_supported_tlds(tld_map: dict) -> None:
    """List TLDs with RDAP + WHOIS support."""
    rdap_tlds = sorted(t for t in tld_map if not t.startswith("xn--"))
    whois_only = sorted(t for t in WHOIS_SERVERS if t not in tld_map and not t.startswith("xn--"))

    print(f"RDAP-supported TLDs ({len(tld_map)} total):\n")
    cols = 8
    for i in range(0, len(rdap_tlds), cols):
        row = rdap_tlds[i: i + cols]
        print("  " + "  ".join(f".{t:<12}" for t in row))

    if whois_only:
        print(f"\nWHOIS-only fallback TLDs ({len(whois_only)}):")
        print("  " + ", ".join(f".{t}" for t in whois_only))

    print(f"\nTLDs not in either list will be tried via rdap.org as a fallback.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_domain.py <domain1> [domain2] ...")
        print("       python check_domain.py --list-tlds")
        print()
        print("Examples:")
        print("  python check_domain.py evermynd.com")
        print("  python check_domain.py cool.dev startup.se app.co mysite.io")
        sys.exit(1)

    bootstrap = load_bootstrap()
    tld_map = build_tld_map(bootstrap)

    if "--list-tlds" in sys.argv:
        print_supported_tlds(tld_map)
        sys.exit(0)

    domains = [d.lower().strip().strip(".") for d in sys.argv[1:] if not d.startswith("--")]

    if not domains:
        print("No domains specified.")
        sys.exit(1)

    print(f"Checking {len(domains)} domain(s)\n")

    for domain in domains:
        result = check_domain(domain, tld_map)
        print_result(result)

        if len(domains) > 1:
            time.sleep(0.3)


if __name__ == "__main__":
    main()
