#date: 2023-11-09T17:07:43Z
#url: https://api.github.com/gists/c7bd2eefba37d875e070835f0750fb9d
#owner: https://api.github.com/users/GoneUp

import argparse
import csv
import requests
import os
import time

# Snipe-IT API information
BASE_URL = ""
API_TOKEN = "**********"
headers = None
 "**********"i "**********"f "**********"  "**********"o "**********"s "**********". "**********"g "**********"e "**********"t "**********"e "**********"n "**********"v "**********"( "**********"" "**********"A "**********"P "**********"I "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"" "**********") "**********"  "**********"! "**********"= "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
    API_TOKEN = "**********"

def get_user_id(email):
    params = {"search": email}
    
    response = requests.get(f"{BASE_URL}/users", headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['total'] > 0:
            user_id = data['rows'][0]['id']
            return user_id

    return None

# Function to check out a seat for a license
def update_user(new_mail, userid):
    payload = {
        "username": new_mail,
        "email": new_mail
    }

    response = requests.patch(f"{BASE_URL}/users/{userid}", json=payload, headers=headers)

    ok = response.json()['status'] == "success"
    return ok

# Read emails from CSV file and process each one
def process_emails(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            time.sleep(1.2) # api rate limit

            email_old = row[0].strip()
            email_new = row[1].strip()

            userid = get_user_id(email_old)
            if not userid:
                print(f"user {email_old} not found")
                continue

            if email_old != "" and email_new != "":
                print(f"--- would update user {userid} from {email_old} to {email_new}")
                result = update_user(email_new, userid)

                
                if result:
                    print(f"updated user {userid} from {email_old} to {email_new}")
                else:
                    print(f"############ Failed to update user: {email_old}")


         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check out seats for emails from a CSV file.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing email addresses.")
    args = parser.parse_args()

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"A "**********"P "**********"I "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"= "**********"= "**********"  "**********"" "**********"" "**********": "**********"
        print("set api token!!!")
        exit(1)

    headers = {"Authorization": "**********"
               'accept': 'application/json',
               'content-type': 'application/json'}
    process_emails(args.csv)
