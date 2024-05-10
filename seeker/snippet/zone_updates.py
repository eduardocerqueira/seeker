#date: 2024-05-10T16:54:40Z
#url: https://api.github.com/gists/5b459923d887ec8dfde51a0d529a2242
#owner: https://api.github.com/users/jvanasco

"""
Cloudflare Zone Updates - migrate an IP address across your account, and enable CT monitoring

Usage:
    This expects the Cloudflare object to be configured with an API token
    as-is, this can be run if the token is exported into the environment
    otherwise, the CF object must be manually created and use an api token
    *api keys will not work*
"""
# pypi
import CloudFlare
import requests


# ==============================================================================


def migrate_ip_address(
    cf: CloudFlare.CloudFlare,
    ip_old: str,
    ip_new: str,
    list_all: bool = False,
):
    zones = cf.zones.get()
    print("Cloudflare Zones:")
    for zone in zones:
        zone_id = zone["id"]
        zone_name = zone["name"]
        print("zone_id=%s zone_name=%s" % (zone_id, zone_name))

        try:
            dns_records = cf.zones.dns_records.get(zone_id)
        except CloudFlare.exceptions.CloudFlareAPIError as e:
            exit("/zones/dns_records.get %d %s - api call failed" % (e, e))

        # then all the DNS records for that zone
        for dns_record in dns_records:
            r_name = dns_record["name"]
            r_type = dns_record["type"]
            r_value = dns_record["content"]
            r_id = dns_record["id"]
            modification_needed: bool = r_value == ip_old
            if list_all or modification_needed:
                print("\t", r_id, r_name, r_type, r_value)
            if modification_needed:
                print("\t", "modifying: %s" % r_id)
                dns_record["content"] = ip_new
                dns_result = cf.zones.dns_records.patch(zone_id, r_id, data=dns_record)


def enable_ct_monitoring(
    cf: CloudFlare.CloudFlare,
):
    _api_url = "https://api.Cloudflare.com/client/v4/zones/%s/ct/alerting"
    # prep requests, as the cloudflare library doesn't handle this nicely
    s = requests.Session()
    s.headers.update(
        {
            "Content-Type": "application/json",
            "Authorization": "**********"
        }
    )

    zones = cf.zones.get()
    print("Cloudflare Zones:")
    for zone in zones:
        zone_id = zone["id"]
        zone_name = zone["name"]
        print("zone_id=%s zone_name=%s" % (zone_id, zone_name))

        alerting_config = s.get(_api_url % zone_id)
        alerting_config_json = alerting_config.json()
        enabled = True if alerting_config_json["result"]["enabled"] else False
        print("\t", "enabled" if enabled else "!!!! DISABLED")
        if not enabled:
            print("\t", "Updating...")
            updated_config = s.patch(_api_url % zone_id, data='{"enabled":true}')
            if updated_config.status_code == 200:
                print("\t", "Success")
            else:
                print("\t", "Failure")


if __name__ == "__main__":
    cf = CloudFlare.CloudFlare()
    migrate_ip_address(cf, ip_old="OLD", ip_new="NEW")
    enable_ct_monitoring(cf)
le_ct_monitoring(cf)
