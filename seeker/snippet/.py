#date: 2025-04-24T17:00:16Z
#url: https://api.github.com/gists/ac6194e47235a89dfadcbeb65a1e1950
#owner: https://api.github.com/users/roncodessmthly

#!/usr/bin/env python3
"""
Script to fetch app information from Google Play Store API
"""
import requests
import json
import os
import argparse
from google.oauth2 import service_account
from googleapiclient.discovery import build

##**flag={Shawcriesalot@999}

def setup_api_client(credentials_file):
    """Setup and authenticate with the Google Play Developer API"""
    credentials = service_account.Credentials.from_service_account_file(
        credentials_file,
        scopes=['https://www.googleapis.com/auth/androidpublisher']
    )
    
    return build('androidpublisher', 'v3', credentials=credentials)

def get_app_info(service, package_name):
    """Get information about a specific app"""
    try:
        # Get app details
        app_details = service.edits().insert(body={}, packageName=package_name).execute()
        edit_id = app_details['id']
        
        # Get app listing information
        listings = service.edits().listings().list(
            packageName=package_name,
            editId=edit_id
        ).execute()
        
        # Get app statistics
        stats = service.reviews().list(
            packageName=package_name,
            maxResults=5
        ).execute()
        
        # Get app ratings
        ratings = service.reviews().get(
            packageName=package_name
        ).execute()
        
        return {
            "listings": listings,
            "reviews": stats,
            "ratings": ratings
        }
    except Exception as e:
        return {"error": str(e)}

def get_random_app(service):
    """Get information about a random popular app"""
    # This is a limited functionality as Google doesn't provide a direct "random app" API
    # Instead, we'll use a small list of popular apps to select from
    popular_apps = [
        "com.instagram.android",
        "com.spotify.music",
        "com.whatsapp",
        "com.facebook.katana",
        "com.google.android.youtube",
        "com.netflix.mediaclient",
        "com.amazon.mShop.android.shopping",
        "com.tiktok.kitty",
        "com.supercell.clashofclans",
        "com.discord"
    ]
    
    import random
    package_name = random.choice(popular_apps)
    return get_app_info(service, package_name)

def main():
    parser = argparse.ArgumentParser(description='Fetch Google Play Store app information')
    parser.add_argument('--credentials', required=True, help='Path to service account credentials JSON file')
    parser.add_argument('--package', help='Package name of the app to fetch info about')
    parser.add_argument('--random', action='store_true', help='Fetch info about a random popular app')
    
    args = parser.parse_args()
    
    service = setup_api_client(args.credentials)
    
    if args.random:
        result = get_random_app(service)
    elif args.package:
        result = get_app_info(service, args.package)
    else:
        parser.print_help()
        return
    
    # Print results in a formatted JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()