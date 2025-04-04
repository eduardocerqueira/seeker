#date: 2025-04-04T17:00:29Z
#url: https://api.github.com/gists/8bd17a6a539b92819ca0a439c5c57b1b
#owner: https://api.github.com/users/rajat-peloton

import json
import csv
import argparse
from collections import defaultdict

'''
python extract_aws_resources_cli.py --input plan.json --output report.csv
python extract_aws_resources_cli.py --input plan.json --output confluence_table.txt --format confluence
'''

def extract_resources_and_access(plan):
    s3_types = ["aws_s3_bucket"]
    db_types = ["aws_dynamodb_table", "aws_rds_cluster", "aws_rds_instance", "aws_db_instance", "aws_redshift_cluster"]
    memcache_types = ["aws_elasticache_parameter_group"]
    kafka_types = ["aws_msk_cluster"]
    additional_services = [
        "aws_kinesis_stream", "aws_sqs_queue",
        "aws_opensearch_domain", "aws_elasticsearch_domain",
        "aws_iam_role"
    ]
    secret_prefix = "arn: "**********":secretsmanager"
    all_supported_types = set(s3_types + db_types + memcache_types + kafka_types + additional_services)

    resource_map = {}
    access_map = defaultdict(list)

    for resource in plan.get("resources", []):
        res_type = resource.get("type")
        if res_type in all_supported_types:
            for instance in resource.get("instances", []):
                attrs = instance.get("attributes", {})
                name = attrs.get("name") or attrs.get("bucket") or attrs.get("id")
                arn = attrs.get("arn", f"{res_type}:{name}")
                resource_map[arn] = (res_type, name)

    for resource in plan.get("resources", []):
        if resource.get("type") == "aws_iam_policy_document":
            for instance in resource.get("instances", []):
                statements = instance.get("attributes", {}).get("statement") or []
                for statement in statements:
                    actions = statement.get("actions", [])
                    resources = statement.get("resources", [])
                    principals = statement.get("principals", [])
                    role_names = []
                    for principal in principals:
                        identifiers = principal.get("identifiers", [])
                        role_names.extend(identifiers)
                    for res in resources:
                        if any(s in res for s in [
                            ": "**********":::", ":dynamodb:", ":elasticache:", ":secretsmanager:",
                            ":kafka:", ":rds:", ":redshift:", ":sqs:", ":kinesis:", ":opensearch:"
                        ]):
                            access_map[res].extend(role_names)

    combined = []
    for arn, (rtype, name) in resource_map.items():
        roles = access_map.get(arn, [])
        combined.append({
            "AWS Resource Type": rtype,
            "Name": name,
            "ARN": arn,
            "IAM Roles with Access": ", ".join(set(roles))
        })

    for res, roles in access_map.items():
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"r "**********"e "**********"s "**********". "**********"s "**********"t "**********"a "**********"r "**********"t "**********"s "**********"w "**********"i "**********"t "**********"h "**********"( "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"p "**********"r "**********"e "**********"f "**********"i "**********"x "**********") "**********"  "**********"a "**********"n "**********"d "**********"  "**********"r "**********"e "**********"s "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"o "**********"u "**********"r "**********"c "**********"e "**********"_ "**********"m "**********"a "**********"p "**********": "**********"
            combined.append({
                "AWS Resource Type": "**********"
                "Name": "**********":secret:")[-1],
                "ARN": res,
                "IAM Roles with Access": ", ".join(set(roles))
            })

    return combined

def write_csv(data, output_path):
    with open(output_path, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["AWS Resource Type", "Name", "ARN", "IAM Roles with Access"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def write_confluence_table(data, output_path):
    with open(output_path, mode="w") as f:
        headers = ["AWS Resource Type", "Name", "ARN", "IAM Roles with Access"]
        f.write("|| " + " || ".join(headers) + " ||\n")
        for row in data:
            line = " | ".join(row[h] for h in headers)
            f.write(f"| {line} |\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract AWS resources and IAM access from a Terraform plan JSON file."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to Terraform plan in JSON format (from terraform show -json)")
    parser.add_argument("--output", "-o", required=True, help="Output file path (.csv or .txt)")
    parser.add_argument("--format", "-f", choices=["csv", "confluence"], default="csv", help="Output format (csv or confluence)")
    args = parser.parse_args()

    # Load JSON
    try:
        with open(args.input, "r") as f:
            plan = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load input file: {e}")
        return

    results = extract_resources_and_access(plan)

    # Write output
    try:
        if args.format == "csv":
            write_csv(results, args.output)
        else:
            write_confluence_table(results, args.output)

        print(f"✅ Extracted {len(results)} resources to {args.output} in {args.format.upper()} format")
    except Exception as e:
        print(f"❌ Failed to write output: {e}")


if __name__ == "__main__":
    main()
