#date: 2023-05-17T16:49:53Z
#url: https://api.github.com/gists/39efec498055a042681550ed02346185
#owner: https://api.github.com/users/sgviking

#!/usr/bin/env python3

import argparse
import configparser
import json
import subprocess
import sys
import os
import requests


VERSION = '1.0.0'

def get_subaccounts(profile):
    args = ["lacework", "account", "list", "--json", "-p", profile]
    results = subprocess.run(args, capture_output=True, text=True, check=False)
    return json.loads(results.stdout)


def get_accounts(profile, subaccount, csp):
    args = ["lacework", "compliance", csp, "list", "--json", "-p", profile, "--subaccount", subaccount]
    results = subprocess.run(args, capture_output=True, text=True, check=False)
    # Azure API returns an empty string if no accounts (different than AWS and GCP)
    if results.stdout == "":
        return {'azure_subscriptions': []}
    return json.loads(results.stdout)


 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"p "**********"r "**********"o "**********"f "**********"i "**********"l "**********"e "**********") "**********": "**********"
    # Read in lacework cli config file and pull details for specified profile
    home = os.path.expanduser('~')
    config = configparser.ConfigParser()
    config.read(home + "/.lacework.toml")
    if not config.has_section(profile):
        return None
    # Use API key and secret to get access token / bearer token
    api_key = config[profile]['api_key'].strip('"')
    api_secret = "**********"
    account = config[profile]['account'].strip('"')
    url = f'https: "**********"
    headers = {'Content-Type': "**********": api_secret}
    data = {'keyId': api_key, 'expiryTime': 36000}
    results = requests.post(url, headers=headers, data=json.dumps(data))
    token = "**********"
    return account, token


 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"r "**********"e "**********"p "**********"o "**********"r "**********"t "**********"s "**********"( "**********"a "**********"c "**********"c "**********"o "**********"u "**********"n "**********"t "**********", "**********"  "**********"s "**********"u "**********"b "**********"a "**********"c "**********"c "**********"o "**********"u "**********"n "**********"t "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"r "**********"e "**********"p "**********"o "**********"r "**********"t "**********", "**********"  "**********"p "**********"r "**********"i "**********"m "**********"a "**********"r "**********"y "**********"_ "**********"i "**********"d "**********", "**********"  "**********"s "**********"e "**********"c "**********"o "**********"n "**********"d "**********"a "**********"r "**********"y "**********"_ "**********"i "**********"d "**********"= "**********"N "**********"o "**********"n "**********"e "**********") "**********": "**********"
    print(f"Getting {report} report for IDs {primary_id}, {secondary_id} in sub-account {subaccount}")
    headers = {'Content-Type': "**********":f'Bearer {token}', 'Account-Name': subaccount}
    url = f'https://{account}.lacework.net/api/v2/Reports?primaryQueryId={primary_id}&format=json&reportType={report}'
    if secondary_id:
        url = f'{url}&secondaryQueryId={secondary_id}'
    results = requests.get(url, headers=headers)
    try:
        return results.json()['data'][0]['summary'][0]
    except:
        return {}


def save_reports(output, reports):
    os.makedirs(output, mode = 0o755, exist_ok = True)
    with open(f'{output}/reports.json', 'w') as file:
        file.write(json.dumps(reports))


def summary(reports, details=False):
    for subaccount in reports:
        csps = {'aws': 'Accounts', 'gcp': 'Projects', 'azure': 'Subscriptions'}
        print(f'{subaccount} -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        for csp in csps:
            rcount = 0
            rfailed = 0
            rexceptions = 0
            not_compliant = 0
            compliant = 0
            highsev = 0
            count = 0

            accounts = reports[subaccount].get(csp, None)
            if not accounts:
                continue
            for account_id, report in reports[subaccount][csp].items():
                total_policies = report.get('NUM_RECOMMENDATIONS', None)
                if total_policies is None:
                    continue
                count = count + 1
                not_compliant = not_compliant + report['NUM_NOT_COMPLIANT']
                compliant = compliant + report['NUM_COMPLIANT']
                highsev = highsev + report['NUM_SEVERITY_1_NON_COMPLIANCE'] + report['NUM_SEVERITY_2_NON_COMPLIANCE']
                rcount = rcount + report['ASSESSED_RESOURCE_COUNT']
                rfailed = rfailed + report['VIOLATED_RESOURCE_COUNT']
                rexceptions = rexceptions + report['SUPPRESSED_RESOURCE_COUNT']
                account_highsev = round((float(report['NUM_SEVERITY_1_NON_COMPLIANCE'] + report['NUM_SEVERITY_2_NON_COMPLIANCE']) / float(total_policies)) * 100, 1)
                account_compliant = round((float(report['NUM_COMPLIANT']) / total_policies) * 100, 1)
                if details:
                    print(f' {account_id}\tCompliant: {account_highsev}%\tHigh severity: {account_compliant}%')
            if count < 1:
                continue

            total = total_policies * count
            percent_compliant = round((float(compliant) / float(total)) * 100, 1)
            percent_highsev = round((float(highsev) / float(total)) * 100, 1)
            print(f' {csp.upper()} {csps[csp]}: {count}\tRecommendations: {total_policies}\tResources: {rcount}\tFailed: {rfailed}\tCompliant: {percent_compliant}%\tHigh severity: {percent_highsev}%')


def parse_args():
    parser = argparse.ArgumentParser(description=f'Pull Lacework compliance reports across multiple sub-accounts.\nVersion: {VERSION}')
    parser.add_argument('-c', '--cache', action='store_true', help='Use the previous API pull in output/')
    parser.add_argument('-d', '--details', action='store_true', help='Show detailed breakdown of each account/project/subscription.')
    parser.add_argument('-p', '--profile', default='default', help='Specify profile to use from lacework CLI configuration. Defaults to \'default\'.')
    parser.add_argument('-o', '--output', default='output', help='Output directory for storing API calls. Defaults to \'output\'')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cache:
        try:
            with open(f'{args.output}/reports.json', 'r') as file:
                reports = json.loads(file.read())
        except:
            print('Error opening {args.output} directory. Re-run without --cache option')
            sys.exit(1)
    else:
        account, token = "**********"
        subaccounts = get_subaccounts(args.profile)
        reports = {}
        for subaccount in subaccounts[0]['accounts']:
            reports[subaccount['accountName']] =  {'aws': {}, 'gcp': {}, 'azure': {}}
            aws_accounts = get_accounts(args.profile, subaccount['accountName'], 'aws')
            for record in aws_accounts['aws_accounts']:
                reports[subaccount['accountName']]['aws'][record['account_id']] = "**********"
            gcp_accounts = get_accounts(args.profile, subaccount['accountName'], 'gcp')
            for record in gcp_accounts['gcp_projects']:
                reports[subaccount['accountName']]['gcp'][record['organization_id'] + '/' + record['project_id']] = "**********"
            azure_accounts = get_accounts(args.profile, subaccount['accountName'], 'azure')
            for record in azure_accounts['azure_subscriptions']:
                reports[subaccount['accountName']]['azure'][record['tenant_id'] + '/' + record['subscription_id']] = "**********"
        save_reports(args.output, reports)
    summary(reports, args.details)


if __name__ == "__main__":
    main()