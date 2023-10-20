#date: 2023-10-20T17:08:57Z
#url: https://api.github.com/gists/15c4d127b941345f0ad2ce75db93e962
#owner: https://api.github.com/users/zoharbabin

import argparse
import logging
from KalturaClient import KalturaClient, KalturaConfiguration
from KalturaClient.Plugins.Core import KalturaAppToken, KalturaAppTokenFilter, KalturaFilterPager, KalturaSessionType
from KalturaClient.exceptions import KalturaException
import hashlib
import json

# Custom logger class
class KalturaLogger:
    def __init__(self):
        self.logger = logging.getLogger('KalturaClient')
        logging.basicConfig(level=logging.DEBUG)

    def log(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"t "**********"a "**********"r "**********"t "**********"_ "**********"a "**********"p "**********"p "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"s "**********"e "**********"s "**********"s "**********"i "**********"o "**********"n "**********"( "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********", "**********"  "**********"p "**********"a "**********"r "**********"t "**********"n "**********"e "**********"r "**********"_ "**********"i "**********"d "**********", "**********"  "**********"a "**********"p "**********"p "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********", "**********"  "**********"a "**********"p "**********"p "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"v "**********"a "**********"l "**********"u "**********"e "**********") "**********": "**********"
    # Start an unprivileged session using session.startWidgetSession
    widget_id = f"_{partner_id}"  # Replace with your actual Partner ID
    unprivileged_ks_response = client.session.startWidgetSession(widget_id)
    unprivileged_ks = unprivileged_ks_response.ks  # Extracting the ks from the response object
    
    if not unprivileged_ks:
        print("Failed to get an unprivileged ks.")
        return None
    
    client.setKs(unprivileged_ks)

    # Compute the Hash
    hash_string = "**********"

    # Start the App Token Session using appToken.startSession
    app_token_session = "**********"
    privileged_ks = "**********"

    # Set the new KS
    client.setKs(privileged_ks)

    return privileged_ks

def build_uri_privilege(list_actions):
    uris = []
    for action in list_actions:
        if '*' in action:
            uri = f"/api_v3/service/{action.replace('.', '/action/').replace('*', '*')}"
        else:
            uri = f"/api_v3/service/{action.replace('.', '/action/')}/"
        uris.append(uri)
    return uris

 "**********"d "**********"e "**********"f "**********"  "**********"l "**********"i "**********"s "**********"t "**********"_ "**********"a "**********"p "**********"p "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********") "**********": "**********"
    filter = "**********"
    pager = KalturaFilterPager()
    
    # Fetch all App Tokens
    result = "**********"
    
    # Print the list of App Tokens
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"a "**********"p "**********"p "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"u "**********"l "**********"t "**********". "**********"o "**********"b "**********"j "**********"e "**********"c "**********"t "**********"s "**********": "**********"
        print(f"App Token ID: "**********"
        print(f"App Token Value: "**********"
        print(f"App Token Description: "**********"
        print("------")

def main():
    parser = "**********"='Manage Kaltura App Tokens.')
    parser.add_argument(
        '-l', '--list', 
        action='store_true', 
        help= "**********"
    )
    parser.add_argument(
        '--actions', 
        type=str, 
        help= "**********"
    )
    parser.add_argument(
        '-u', '--update', 
        type=str, 
        help= "**********"
    )
    parser.add_argument(
        '-a', '--append', 
        action='store_true', 
        help= "**********"
    )
    parser.add_argument(
        '-d', '--description',
        type=str,
        help= "**********"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug printing.'
    )
    
    args = parser.parse_args()
    
    # Check for mandatory 'actions' argument when adding or updating an App Token
    if (args.update or not args.list) and not args.actions:
        print("The --actions argument is mandatory when adding or updating an App Token.")
        return

    list_actions = args.actions.lower().split(',') if args.actions else None

    # Load configuration from JSON file
    with open('config.json', 'r') as f:
        config_data = json.load(f)
    PARTNER_ID = config_data['PARTNER_ID']
    ADMIN_SECRET = "**********"
    SCRIPT_USER_ID = config_data['SCRIPT_USER_ID']
    ADMIN_SESSION_EXPIRY = config_data['ADMIN_SESSION_EXPIRY']
    KALTURA_SERVICE_URL = config_data['KALTURA_SERVICE_URL']

    # Initialize the Kaltura client
    config = KalturaConfiguration(PARTNER_ID)
    config.serviceUrl = KALTURA_SERVICE_URL
    if args.debug:
        kaltura_logger = KalturaLogger()
        config.setLogger(kaltura_logger)
    client = KalturaClient(config)
    client.requestHeaders = {
        'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    }
    
    # Start an admin-level Kaltura session using your Admin Secret
    ks = "**********"
    
    client.setKs(ks)
    
    if args.list:
        list_app_tokens(client)
        return  # Exit after listing all tokens
    
    print_text_prefix = ''
    app_token_id = "**********"
    app_token_value = "**********"
    session_privileges = None

    if args.update:
        # Update existing App Token
        try:
            existing_app_token = "**********"
        except KalturaException as e:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"e "**********". "**********"c "**********"o "**********"d "**********"e "**********"  "**********"= "**********"= "**********"  "**********"' "**********"A "**********"P "**********"P "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"I "**********"D "**********"_ "**********"N "**********"O "**********"T "**********"_ "**********"F "**********"O "**********"U "**********"N "**********"D "**********"' "**********": "**********"
                print(f"App Token ID {args.update} not found. Please use a valid ID.")
                return
            else:
                raise e

        if args.append:
            # Create a list of existing URIs by splitting the string on "|"
            existing_uris = "**********"
            
            # Add new URIs to the list
            new_uris = build_uri_privilege(list_actions)
            existing_uris.extend(new_uris)
            
            # Remove duplicates
            unique_uris = list(set(existing_uris))
            
            # Convert back to string
            existing_app_token.sessionPrivileges = "urirestrict: "**********"
        else:
            existing_app_token.sessionPrivileges = "**********"
        
        if args.description:
            existing_app_token.description = "**********"

        result = "**********"
        app_token_id = "**********"
        app_token_value = "**********"
        session_privileges = result.sessionPrivileges
        print_text_prefix = "**********"
    else:
        # Add a new App Token
        app_token = "**********"
        app_token.sessionType = "**********"
        app_token.sessionPrivileges = "**********"
        app_token.hashType = "**********"
        
        if args.description:
            app_token.description = "**********"

        result = "**********"
        app_token_id = "**********"
        app_token_value = "**********"
        session_privileges = result.sessionPrivileges
        print_text_prefix = "**********"

    print(print_text_prefix + f"Value: "**********"
    print(print_text_prefix + f"Privileges: {session_privileges}")

 "**********"  "**********"  "**********"  "**********"  "**********"# "**********"  "**********"G "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"  "**********"a "**********"  "**********"s "**********"a "**********"m "**********"p "**********"l "**********"e "**********"  "**********"K "**********"S "**********"  "**********"f "**********"r "**********"o "**********"m "**********"  "**********"t "**********"h "**********"i "**********"s "**********"  "**********"a "**********"p "**********"p "**********"T "**********"o "**********"k "**********"e "**********"n "**********": "**********"
    token_ks = "**********"
    print(f"Gen KS from this AppToken: "**********"

if __name__ == "__main__":
    main()