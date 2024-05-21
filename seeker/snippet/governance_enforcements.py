#date: 2024-05-21T17:08:16Z
#url: https://api.github.com/gists/0e8230b5c422e9a2c692db50171ae031
#owner: https://api.github.com/users/rssnyder

from os import getenv

import pandas as pd
import numpy as np
from requests import post, put


HARNESS_URL = "app3.harness.io"

PARAMS = {
    "routingId": getenv("HARNESS_ACCOUNT_ID"),
    "accountIdentifier": getenv("HARNESS_ACCOUNT_ID"),
}

HEADERS = {
    "x-api-key": getenv("HARNESS_PLATFORM_API_KEY"),
}


def create_enforcement(
    name: str,
    rule_set_ids: list[str],
    accounts: list[str],
    regions: list[str],
    dry_run: bool = True,
    enabled: bool = False,
    description: str = "created from automation",
    schedule: str = "0 0 0 ? * 0",
    tz: str = "America/Chicago",
):

    print(f"creating {name}")

    resp = post(
        f"https://{HARNESS_URL}/gateway/ccm/api/governance/enforcement",
        headers=HEADERS,
        params=PARAMS,
        json={
            "ruleEnforcement": {
                "name": name,
                "description": description,
                "cloudProvider": "AWS",
                "ruleIds": [],
                "ruleSetIDs": rule_set_ids,
                "executionSchedule": schedule,
                "executionTimezone": tz,
                "isEnabled": enabled,
                "targetAccounts": accounts,
                "targetRegions": regions,
                "isDryRun": dry_run,
            }
        },
    )

    if resp.json().get("status") == "SUCCESS":
        return resp.json().get("data", {}).get("uuid")
    else:
        if "already" in resp.json().get("message"):
            print(f"already exists")

            uuid = get_enforcement(name).get("uuid")

            print(f"found as {uuid}, updating")
            
            resp = put(
                f"https://{HARNESS_URL}/gateway/ccm/api/governance/enforcement",
                headers=HEADERS,
                params=PARAMS,
                json={
                    "ruleEnforcement": {
                        "uuid": uuid,
                        # "name": name,
                        "description": description,
                        "cloudProvider": "AWS",
                        "ruleIds": [],
                        "ruleSetIDs": rule_set_ids,
                        "executionSchedule": schedule,
                        "executionTimezone": tz,
                        "isEnabled": enabled,
                        "targetAccounts": accounts,
                        "targetRegions": regions,
                        "isDryRun": dry_run,
                    }
                },
            )

            return resp.json().get("status", "unknown result")
        else:
            return resp.text


def get_enforcement(name: str):
    resp = post(
        f"https://{HARNESS_URL}/gateway/ccm/api/governance/enforcement/list",
        headers=HEADERS,
        params=PARAMS,
        json={
            "ruleEnforcement": {
                "orderBy": [{"field": "RULE_ENFORCEMENT_NAME", "order": "ASCENDING"}],
                "limit": 10,
                "offset": 0,
                "search": name,
            }
        },
    )

    return resp.json().get("data").pop()


def get_rule_set(name: str) -> dict:
    resp = post(
        f"https://{HARNESS_URL}/gateway/ccm/api/governance/ruleSet/list",
        headers=HEADERS,
        params=PARAMS,
        json={
            "ruleSet": {
                "orderBy": [{"field": "RULE_SET_NAME", "order": "ASCENDING"}],
                "limit": 10,
                "offset": 0,
                "search": name,
            }
        },
    )

    if resp.json().get("status") == "SUCCESS":
        return resp.json().get("data", {}).get("ruleSet", []).pop()
    else:
        return {}


def clean_account_id(old_id: str) -> str:
    return str(int(old_id)).zfill(12)


if __name__ == "__main__":

    # load in excel file of dish aws account
    file = pd.ExcelFile("BF-GF-OTT-AWS-Accounts-27feb2024.xlsx")

    # name of the rule set that holds the rules that trigger pipelines
    pipeline_rules = get_rule_set("tkvg-asset-governance-approval-pipeline-review-for-testing")
    
    # name of the rule set that holds the rules that delete resources, for dry run and cost estimation
    delete_rules = get_rule_set("tkvg-cmm-delete-rules-review-active-for-testing")

    # name of the rule set that holds the rules for ec2 auto stopping
    ec2_autostopping_rules = get_rule_set("tkvg-ec2-autostopping-active-review")

    # name of the rule set that holds the rules for ec2 auto stopping
    rds_autostopping_rules = get_rule_set("tkvg-rds-autostopping-active-review")

    # for every sheet in the csv
    for sheet in file.sheet_names:

        # skip these sheets for now
        if sheet.strip() in ["Master Accounts", "Ott sling"]:
            continue

        # show the sheet we are working on
        print(sheet)

        # lists for each account type
        sbx_accts = []
        np_accts = []

        # for every account in the sheet
        for _, row in file.parse(sheet).iterrows():

            # make sure the account is a linked account (member not master)
            if (row.get("Type") == "Linked Account") and (not np.isnan(row.get("Linked account"))):

                # rules for finding sandbox account
                if row.get("Linked account name").strip().endswith("sbx"):
                    sbx_accts.append(clean_account_id(row.get("Linked account")))
                
                # rules for finding nonprod accounts
                elif row.get("Linked account name").strip().endswith("np"):
                    np_accts.append(clean_account_id(row.get("Linked account")))

        # create sandbox enforcement for the approval pipelines
        resp = create_enforcement(
            f"{sheet} All Sandbox Pipeline Approvals",
            [pipeline_rules.get("uuid")],
            sbx_accts,
            ["us-west-2"],
            dry_run=False,
            description=f"runs pipeline rules on all {sheet} sandbox accounts.",
        )

        print(resp)

        # create sandbox enforcement for the ec2 autostopping
        resp = create_enforcement(
            f"{sheet} All Sandbox EC2 Autostopping",
            [ec2_autostopping_rules.get("uuid")],
            sbx_accts,
            ["us-west-2"],
            dry_run=False,
            description=f"runs rules on {sheet} to gather info on ec2/rds instances and trigger autostopping pipelines.",
        )

        print(resp)

        # create sandbox enforcement for the rds autostopping
        resp = create_enforcement(
            f"{sheet} All Sandbox RDS Autostopping",
            [rds_autostopping_rules.get("uuid")],
            sbx_accts,
            ["us-west-2"],
            dry_run=False,
            description=f"runs rules on {sheet} to gather info on ec2/rds instances and trigger autostopping pipelines.",
        )

        print(resp)

        # create sandbox enforcement for the dry run rules
        resp = create_enforcement(
            f"DRYDRUN {sheet} All Sandbox DELETE RULES",
            [pipeline_rules.get("uuid")],
            sbx_accts,
            ["us-west-2"],
            dry_run=True,
            description=f"runs pipeline rules on all {sheet} sandbox accounts.",
        )

        print(resp)

        # create non-prod enforcement for the approval pipelines
        resp = create_enforcement(
            f"{sheet} All NonProd Pipeline Approvals",
            [pipeline_rules.get("uuid")],
            np_accts,
            ["us-west-2"],
            dry_run=False,
            description=f"runs pipeline rules on all {sheet} non-prod accounts.",
        )

        print(resp)

        # create non-prod enforcement for the ec2 autostopping
        resp = create_enforcement(
            f"{sheet} All NonProd EC2 Autostopping",
            [ec2_autostopping_rules.get("uuid")],
            np_accts,
            ["us-west-2"],
            dry_run=False,
            description=f"runs rules on {sheet} to gather info on ec2/rds instances and trigger autostopping pipelines.",
        )

        print(resp)

        # create non-prod enforcement for the rds autostopping
        resp = create_enforcement(
            f"{sheet} All NonProd RDS Autostopping",
            [rds_autostopping_rules.get("uuid")],
            np_accts,
            ["us-west-2"],
            dry_run=False,
            description=f"runs rules on {sheet} to gather info on ec2/rds instances and trigger autostopping pipelines.",
        )

        print(resp)

        # create non-prod enforcement for the dry run rules
        resp = create_enforcement(
            f"DRYDRUN {sheet} All NonProd DELETE RULES",
            [pipeline_rules.get("uuid")],
            np_accts,
            ["us-west-2"],
            dry_run=True,
            description=f"runs pipeline rules on all {sheet} non-prod accounts.",
        )

        print(resp)
