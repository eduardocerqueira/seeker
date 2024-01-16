#date: 2024-01-16T16:46:07Z
#url: https://api.github.com/gists/1376e10e3619e9c65a6747261752af9c
#owner: https://api.github.com/users/ThomasLachaux

# reporter/report.py
import boto3
import jinja2

sh = boto3.client("securityhub")

# Replace with your accounts ids
prod_accounts = ["012345678901", "123456789012"]
staging_accounts = ["345678901234", "456789012345"]
tooling_accounts = ["567890123456"]

all_accounts = prod_accounts + staging_accounts + tooling_accounts

def get_findings(filters):
    pages = sh.get_paginator("get_findings").paginate(Filters=filters, PaginationConfig={"PageSize": 100})

    findings = []

    for page in pages:
        for finding in page["Findings"]:
            findings.append(finding)

    return findings

def get_enabled_standards():
    response = sh.get_enabled_standards()["StandardsSubscriptions"]
    standards = [x["StandardsArn"].split("::")[1] for x in response]

    return standards


class FindingSet:
    def __init__(self, findings):
        self.findings = findings

    def filter_by_accounts(self, *accounts):
        self.findings = [x for x in self.findings if x["AwsAccountId"] in accounts]
        return self

    def filter_by_standard(self, standard):
        self.findings = [
            x for x in self.findings if standard in [y["StandardsId"] for y in x["Compliance"]["AssociatedStandards"]]
        ]
        return self

    def compute_score(self):
        controls = {}

        for finding in self.findings:
            if finding["Workflow"]["Status"] == "SUPPRESSED":
                continue

            if finding["Workflow"]["Status"] == "NEW":
                controls[finding["Compliance"]["SecurityControlId"]] = False
                continue

            if finding["Compliance"]["SecurityControlId"] not in controls:
                controls[finding["Compliance"]["SecurityControlId"]] = True

        score = sum(controls.values()) / len(controls)
        score = round(score * 100)

        score_type = "score-low"

        if score >= 50:
            score_type = "score-neutral"

        if score >= 80:
            score_type = "score-high"

        score = str(score) + "%"
        score = f'<span class="{score_type}">{score}</span>'

        return score


def display_standard_name(standard):
    return standard.split("/")[1].replace("-", " ")


def render():
    enabled_standards = get_enabled_standards()

    filters = {
        "RecordState": eq("ACTIVE"),
        "ComplianceAssociatedStandardsId": eq(*get_enabled_standards()),
    }
    all_findings = get_findings(filters)

    standards_report = []
    for standard in enabled_standards:
        print(standard)
        findings = {
            "prod": FindingSet(all_findings)
            .filter_by_accounts(*p_accounts)
            .filter_by_standard(standard)
            .compute_score(),
            "oop": FindingSet(all_findings)
            .filter_by_accounts(*oop_accounts)
            .filter_by_standard(standard)
            .compute_score(),
            "tooling": FindingSet(all_findings)
            .filter_by_accounts(*tooling_accounts)
            .filter_by_standard(standard)
            .compute_score(),
            "global": FindingSet(all_findings).filter_by_standard(standard).compute_score(),
        }

        standards_report.append({"name": display_standard_name(standard), "findings": findings})

    standards_report.append(
        {
            "name": "Global",
            "findings": {
                "prod": FindingSet(all_findings).filter_by_accounts(*p_accounts).compute_score(),
                "oop": FindingSet(all_findings).filter_by_accounts(*oop_accounts).compute_score(),
                "tooling": FindingSet(all_findings).filter_by_accounts(*tooling_accounts).compute_score(),
                "global": FindingSet(all_findings).compute_score(),
            },
        }
    )

    with open("template.html") as f:
        template = jinja2.Template(f.read())

    return template.render(standards=standards_report)


if __name__ == "__main__":
    template = render()
    with open("out.html", "w") as f:
        f.write(template)