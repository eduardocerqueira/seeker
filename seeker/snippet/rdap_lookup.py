#date: 2025-06-17T16:56:35Z
#url: https://api.github.com/gists/b2d79f561c60e1236de92e9836998285
#owner: https://api.github.com/users/tn3w

import urllib.request
import urllib.error
import json
import time
from typing import Dict, Optional, Any


def get_rdap_info(ip_address: str) -> Optional[Dict[str, Any]]:
    """
    Query RDAP (Registration Data Access Protocol) for IP information.
    RDAP is the modern replacement for WHOIS.

    Args:
        ip_address (str): IP address to look up

    Returns:
        dict: Detailed network information including organization, network range,
             registration dates, status, and contact information
    """
    url = f"https://rdap.arin.net/registry/ip/{ip_address}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/rdap+json"})

        with urllib.request.urlopen(req, timeout=5) as response:
            content = response.read().decode("utf-8")
            data = json.loads(content)

        result: Dict[str, Any] = {
            "source": "RDAP",
            "network": None,
            "organization": None,
            "abuse_contacts": [],
            "admin_contacts": [],
            "tech_contacts": [],
            "status": [],
            "country": None,
            "registration_date": None,
            "last_changed_date": None,
            "description": [],
            "remarks": [],
            "type": None,
            "ip_version": None,
            "network_range": None,
        }

        if "cidr0_cidrs" in data:
            cidrs = data.get("cidr0_cidrs", [])
            if cidrs and len(cidrs) > 0:
                result["network"] = (
                    cidrs[0].get("v4prefix", "") + "/" + str(cidrs[0].get("length", ""))
                )

        result["organization"] = data.get("name")
        result["status"] = data.get("status", [])
        result["country"] = data.get("country")
        result["type"] = data.get("type")
        result["ip_version"] = data.get("ipVersion")

        if "startAddress" in data and "endAddress" in data:
            result["network_range"] = f"{data['startAddress']} - {data['endAddress']}"

        events = data.get("events", [])
        for event in events:
            if event.get("eventAction") == "registration":
                result["registration_date"] = event.get("eventDate")
            elif event.get("eventAction") == "last changed":
                result["last_changed_date"] = event.get("eventDate")

        remarks = data.get("remarks", [])
        for remark in remarks:
            if remark.get("title") == "description":
                result["description"].extend(remark.get("description", []))
            else:
                result["remarks"].extend(remark.get("description", []))

        entities = data.get("entities", [])
        for entity in entities:
            roles = entity.get("roles", [])
            vcard = entity.get("vcardArray", [])

            contact_info: Dict[str, str] = {}
            if len(vcard) > 1 and isinstance(vcard[1], list):
                for field in vcard[1]:
                    if field[0] == "fn":
                        contact_info["name"] = field[3]
                    elif field[0] == "email":
                        contact_info["email"] = field[3]
                    elif field[0] == "tel":
                        contact_info["phone"] = field[3]
                    elif field[0] == "adr" and "label" in field[1]:
                        contact_info["address"] = field[1]["label"]

            if contact_info:
                if "abuse" in roles:
                    result["abuse_contacts"].append(contact_info)
                if "administrative" in roles:
                    result["admin_contacts"].append(contact_info)
                if "technical" in roles:
                    result["tech_contacts"].append(contact_info)

        return result
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"Error querying RDAP: {e}")

    return None

if __name__ == "__main__":
    ip = "1.1.1.1"

    start_time = time.time()
    rdap_data = get_rdap_info(ip)
    print(f"Time taken: {time.time() - start_time} seconds")
    print(rdap_data)
