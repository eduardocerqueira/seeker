#date: 2025-03-24T16:42:41Z
#url: https://api.github.com/gists/b48dd1f116fdf60293c6be891ac964f1
#owner: https://api.github.com/users/tim-snyk

import argparse
import os
import urllib
import traceback
import requests
import time
import dateutil.parser
import datetime
from urllib.parse import urlparse, quote
import concurrent.futures

'''
Description: A script to identify and delete stale assets in a Snyk Group or Organization based on age threshold.
Author:  Tim Gowan (tim.gowan@snyk.io)
'''
snykToken = "**********"
url = "https://api.snyk.io"
apiVersion = "2024-10-15"  # Set the API version.
tries = 8  # Number of retries
delay = 1  # Delay between retries
backoff = 2  # Backoff factor
limit = 100 # Pagination limit
max_workers = 20  # Number of concurrent workers

class Group:
    def __init__(self, groupId):
        # string
        self.id = groupId
        self.orgs = self.listOrgsInGroup()

    def listOrgsInGroup(self, orgName=None, orgSlug=None):
        '''
        # GET /rest/groups/{group_id}/orgs
        '''
        uri = f"/rest/groups/{self.id}/orgs?version={apiVersion}"
        if limit:
            uri += f"&limit={limit}"
        if orgName:
            uri += f"&name={orgName}"
        if orgSlug:
            uri += f"&slug={orgSlug}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "**********"
        }
        response = requests.get(
            url + uri,
            headers=headers
        )
        organizations = []
        while response.status_code == 200:
            response_json = response.json()
            if "data" in response_json:
                for org in response_json['data']:
                    organizations.append(Organization(org['id']))
            uri = response_json["links"].get("next") if "links" in response_json else None
            if uri:
                response = requests.get(url + uri, headers=headers)
            else:
                break
        return organizations

class Organization:
    def __init__(self, orgId):
        self.id = orgId
        if self.id is not None:
            org = self.getOrg()
            self.groupId = org['data']['attributes']['group_id']
            self.name = org['data']['attributes']['name']
            self.slug = org['data']['attributes']['slug']
            self.integrations = self.listIntegrations()
            self.targets = self.listTargetsInOrg()

    def listTargetsInOrg(self, target_name=None, created_gte=None, exclude_empty='false'):
        '''
        # GET /rest/orgs/{org_id}/targets
        '''
        uri = f"/rest/orgs/{self.id}/targets?version={apiVersion}"
        if limit:
            uri += f"&limit={limit}"
        if target_name:
            uri += f"&display_name={target_name}"
        if exclude_empty:
            uri += f"&exclude_empty={exclude_empty}"

        if created_gte:
            created_gte = dateutil.parser.parse(created_gte).astimezone()
            created_gte_url_safe = quote(created_gte.isoformat())
            uri += f"&created_gte={created_gte_url_safe}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "**********"
        }
        try:
            response = requests.get(
                url + uri,
                headers=headers
            )
            response.raise_for_status()
            targets = []
            while response.status_code in [200, 429]:
                if response.status_code == 429:
                    print(f"Rate limit exceeded, retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                response_json = response.json()

                if "data" in response_json:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(Target, self, org['id']) for org in response_json['data']]
                        for future in concurrent.futures.as_completed(futures):
                            targets.append(future.result())
                uri = response_json["links"].get("next") if "links" in response_json else None
                if uri:
                    response = requests.get(url + uri, headers=headers)
                else:
                    break
        except requests.exceptions.RequestException as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            exit(1)
        # Uncomment print statement to debug
        # print(json.dumps(response.json(), indent=4))
        return targets

    def getOrg(self):
        '''
        # GET /rest/orgs/{orgId}?version={apiVersion}
        '''
        uri = f"/rest/orgs/{self.id}?version={apiVersion}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "**********"
        }
        try:
            response = requests.get(
                url + uri,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            exit(1)

    def listIntegrations(self) -> 'integrations':
        '''
        # GET https://api.snyk.io/v1/org/{orgId}/integrations
        '''
        uri = f"/v1/org/{self.id}/integrations"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "**********"
        }
        response = requests.get(
            url + uri,
            headers=headers
        )
        return response.json()

class Target:
    def __init__(self, organization, target_id):
        self.id = target_id
        self.organization = organization
        target = self.getTarget()
        if target is not None:
            self.created_at = target['data']['attributes']['created_at']
            self.display_name = target['data']['attributes']['display_name']
            self.integration_id = target['data']['relationships']['integration']['data']['id']
            self.integration_type = target['data']['relationships']['integration']['data']['attributes']['integration_type']
            created_at_dt = datetime.datetime.strptime(self.created_at, '%Y-%m-%dT%H:%M:%S.%fZ').astimezone()
            now_dt = datetime.datetime.now().astimezone()
            self.age = now_dt - created_at_dt

    def printTarget(self):
        print(f"    Target ID: {self.id}")
        print(f"        Age: {self.age}")
        print(f"        Created At: {self.created_at}")
        print(f"        Display Name: {self.display_name}")
        print(f"        Integration Type: {self.integration_type}")
        print(f"        Organization: {self.organization.name} ({self.organization.id})")

    def getTarget(self):
        # GET /orgs/{org_id}/targets/{target_id}
        uri = f"/rest/orgs/{self.organization.id}/targets/{self.id}?version={apiVersion}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "**********"
        }
        try:
            response = requests.get(
                url + uri,
                headers=headers
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"An unexpected error occurred: {e}")
            print(f"Request URL: {url + uri}")
            traceback.print_exc()
            exit(1)
        # Uncomment print statement to debug
        #print(json.dumps(response.json(), indent=4))
        return response.json()

    def deleteTarget(self):
        # DELETE /orgs/{org_id}/targets/{target_id}
        uri = f"/rest/orgs/{self.organization.id}/targets/{self.id}?version={apiVersion}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': "**********"
        }
        try:
            response = requests.delete(
                url + uri,
                headers=headers
            )
            response.raise_for_status()
            print(f"[DELETED] {target.id}")
        except requests.exceptions.RequestException as e:
            print(f"An unexpected error occurred: {e}")
            print(f"Request URL: {url + uri}")
            traceback.print_exc()
            exit(1)

if __name__ == '__main__':
    # Parsing Command Line Arguments
    parser = argparse.ArgumentParser(
        description='Identify assets in Snyk Group.')
    # Required fields:
    parser.add_argument('--groupId', '-g',
                        type=str,
                        help='Snyk Group ID, groupId of the group to query.',
                        required=False)
    parser.add_argument('--orgId', '-o',
                        type=str,
                        help='Snyk Organization ID, orgId of the organization to query.',
                        required=False)
    parser.add_argument('--ageThreshold', '-t',
                        type=int,
                        help='Age threshold in days. Assets older than this threshold will be considered for deletion.',
                        required=True)
    parser.add_argument('--delete', '-d',
                        action='store_true',
                        help='Default to dry run mode. When this flag is provided, the script will not delete any assets.',
                        required=False)
    args = parser.parse_args()
    if args.delete is not True:
        print("[DRY RUN MODE] No assets will be deleted. Use --delete (-d) flag to delete assets.")
    else:
        print("[DELETE MODE] Assets will be deleted.")
    if args.ageThreshold is not None:
        print(f"[AGE THRESHOLD] {args.ageThreshold} days. Assets older than this threshold will be considered.")

    staleTargets = []
    if args.groupId is not None:
        args.groupId = urllib.parse.quote_plus(args.groupId)
        group = Group(args.groupId)
        for org in group.orgs:
            # Subtract args.ageThreshold as days from current time
            ageThreshold = datetime.datetime.now() - datetime.timedelta(days=args.ageThreshold)
            for target in org.targets:
                if datetime.datetime.strptime(target.created_at, '%Y-%m-%dT%H:%M:%S.%fZ') < ageThreshold:
                    staleTargets.append(target)

    elif args.orgId is not None:
        org = Organization(args.orgId)
        # Subtract args.ageThreshold as days from current time
        ageThreshold = datetime.datetime.now() - datetime.timedelta(days=args.ageThreshold)
        for target in org.targets:
            if datetime.datetime.strptime(target.created_at, '%Y-%m-%dT%H:%M:%S.%fZ') < ageThreshold:
                staleTargets.append(target)
    else:
        print("No group or org ID provided. Please provide either --orgId (-o) or --groupId (-g) parameters with"
              "the corresponding UUID. Exiting.")
    print(f"Found {len(staleTargets)} stale repository assets older than {args.ageThreshold} days.")
    for target in staleTargets:
        target.printTarget()
        if args.delete is True:
            target.deleteTarget()
    exit()
et()
        if args.delete is True:
            target.deleteTarget()
    exit()
