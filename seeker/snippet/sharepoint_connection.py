#date: 2024-08-27T16:59:37Z
#url: https://api.github.com/gists/f22337768cf3a0f9bb74fca1f8c096b0
#owner: https://api.github.com/users/luisdelatorre012

"""
dynaconf settings

# settings.toml
[default]
site_name = "Your SharePoint Site Name"

# .secrets.toml
[default]
client_id = "your_client_id_here"
client_secret = "**********"
tenant_id = "your_tenant_id_here"
"""

import httpx
from msal import ConfidentialClientApplication
from typing import Optional
from dynaconf import Dynaconf

# Initialize Dynaconf
settings = Dynaconf(
    settings_files= "**********"
    environments=True,
    load_dotenv=True,
)

def get_access_token(client_id: "**********": str, tenant_id: str) -> Optional[str]:
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = ConfidentialClientApplication(
        client_id,
        authority=authority,
        client_credential= "**********"
    )
    
    scopes = ["https://graph.microsoft.com/.default"]
    result = "**********"=None)
    
    if not result:
        result = "**********"=scopes)
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"" "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"u "**********"l "**********"t "**********": "**********"
        return result["access_token"]
    else:
        print(result.get("error"))
        print(result.get("error_description"))
        print(result.get("correlation_id"))
        return None

def get_sharepoint_site_id(access_token: "**********": str) -> Optional[str]:
    headers = {
        "Authorization": "**********"
        "Content-Type": "application/json"
    }
    
    url = f"https://graph.microsoft.com/v1.0/sites?search={site_name}"
    
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
    
    if response.status_code == 200:
        sites = response.json().get("value", [])
        if sites:
            return sites[0]["id"]
    
    print(f"Error: {response.status_code}")
    print(response.text)
    return None

def main() -> None:
    # Read configuration from Dynaconf
    client_id = settings.client_id
    client_secret = "**********"
    tenant_id = settings.tenant_id
    site_name = settings.site_name
    
    access_token = "**********"
    
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        site_id = "**********"
        if site_id:
            print(f"Successfully connected to SharePoint site. Site ID: {site_id}")
        else:
            print("Failed to retrieve SharePoint site ID.")
    else:
        print("Failed to acquire access token.")

if __name__ == "__main__":
    main()

