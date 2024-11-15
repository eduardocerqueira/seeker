#date: 2024-11-15T17:06:26Z
#url: https://api.github.com/gists/5ed51de5f3c32548418f35ae7316d62b
#owner: https://api.github.com/users/layaxx

#!/usr/bin/python3
import os
import sys
import requests
from datetime import datetime

encoded_domain_name = "yourdomain_tld"
hostname = "your-desired-subdomain"
contact_address = "replaceme@example.com"


def update_dns_settings(type, new_value):
    req = {
        "hostname": hostname,
        "ttl": 3600,
        "type": "A" if type == "ipv4" else "AAAA",
        "value": new_value,
    }

    headers = {
        "User-Agent": f"DNSUpdate ({contact_address})",
        "Authorization": "**********"
    }

    response = requests.post(
        "https://api.netlify.com/api/v1/dns_zones/"
        + encoded_domain_name
        + "/dns_records",
        data=req,
        headers=headers,
    )

    if response.ok:
        print("Update successful")

        id_path = f"./cache/{type}.old-id.txt"
        try:
            with open(id_path, "r") as f:
                old_id = f.read()
                delete_response = requests.delete(
                    f"https://api.netlify.com/api/v1/dns_zones/{encoded_domain_name}/dns_records/{old_id}",
                    headers=headers,
                )

                if delete_response.ok:
                    print("deleted superseded record")
                else:
                    print("failed to delete old record")
        except FileNotFoundError:
            pass

        try:
            with open(id_path, "w+") as f:
                f.write(response.json()["id"])
                print("noted id of this dns record")
        except:
            pass
    else:
        print("Failed to update")
        print(response)


def update_if_necessary(type, response):
    cache_path = f"./cache/{type}.txt"
    if response.ok:
        address = response.text
        try:
            with open(cache_path, "r") as f:
                old_address = f.read()
        except FileNotFoundError:
            old_address = "cache-not-found"

        if old_address == address:
            print(f"{type} address has not changed")
        else:
            with open(cache_path, "w") as f:
                f.write(address)
                print(f"Updating IPv4 from {old_address} to {address}")
                update_dns_settings(type, address)

    else:
        print("Failed to determine {type} address")


if __name__ == "__main__":
    print(">>> " + str(datetime.now()))

    # make sure that the Netlify auth token exists
    try:
        API_TOKEN = "**********"
    except KeyError:
        print("Failed to get NETLIFY_API_TOKEN from environment variables")
        sys.exit(1)

    # determine public ipv4 and ipv6 addresses
    ipv4_response = requests.get("https://api.ipify.org")
    ipv6_response = requests.get("https://api6.ipify.org")

    update_if_necessary("ipv4", ipv4_response)
    update_if_necessary("ipv6", ipv6_response)
ssary("ipv6", ipv6_response)
