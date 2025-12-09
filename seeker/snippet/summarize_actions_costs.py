#date: 2025-12-09T17:04:32Z
#url: https://api.github.com/gists/98cb03ec43b716b1f8e03bcc091d069c
#owner: https://api.github.com/users/docs-bot

#!/usr/bin/env python3
"""
Actions Cost Summary Script

Processes GitHub Actions billing data from summarized usage reports to generate
cost summaries by SKU (Stock Keeping Unit).

For more information about summarized usage reports, see:
https://docs.github.com/en/billing/reference/billing-reports#summarized-usage-report

Requirements:
    - Python 3.x (uses only standard library modules)

Usage:
    # Generate a single SKU summary file
    python3 summarize_actions_costs.py input_file.csv

    # Generate SKU summary plus per-organization files
    python3 summarize_actions_costs.py input_file.csv --by-org

Output:
    - input_file_sku.csv: Overall SKU summary with total costs and free minutes
    - input_file.organization_name.csv: Per-organization summaries (with --by-org)

    Each output file contains:
    - sku: The SKU identifier (only those starting with 'actions_')
    - total_net_amount: Sum of all costs for this SKU
    - free_minutes_quantity: Sum of quantity used when net_amount is 0

Notes:
    - Only processes rows where product is 'actions'
    - Organization names are sanitized for safe filenames
    - Validates required CSV columns before processing
    - Maximum of 1000 organizations to prevent resource exhaustion
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def process_billing_data(input_file, by_organization=False):
    """
    Process a billing CSV file and summarize costs by SKU.

    Args:
        input_file: Path to the input CSV file
        by_organization: If True, group by organization; otherwise overall summary

    Returns:
        Dictionary with summaries (by org if specified, otherwise overall by SKU)
    """
    sku_data = defaultdict(lambda: {
        'total_cost': 0.0,
        'free_minutes': 0.0
    })
    org_sku_data = defaultdict(lambda: defaultdict(lambda: {
        'total_cost': 0.0,
        'free_minutes': 0.0
    }))

    required_columns = {'product', 'sku', 'net_amount', 'quantity'}
    if by_organization:
        required_columns.add('organization')

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Validate required columns exist
        if not required_columns.issubset(reader.fieldnames or []):
            missing = required_columns - set(reader.fieldnames or [])
            raise ValueError(f"Missing required columns: {missing}")

        for row_num, row in enumerate(reader, start=2):  # start=2 accounts for header
            try:
                product = row['product']

                # Only process actions product
                if product != 'actions':
                    continue

                sku = row['sku']

                # Only process SKUs that start with 'actions_'
                if not sku.startswith('actions_'):
                    continue

                net_amount = float(row['net_amount'])
                quantity = float(row['quantity'])

                if by_organization:
                    organization = row['organization']
                    org_sku_data[organization][sku]['total_cost'] += net_amount
                    if net_amount == 0:
                        org_sku_data[organization][sku]['free_minutes'] += quantity
                else:
                    sku_data[sku]['total_cost'] += net_amount
                    if net_amount == 0:
                        sku_data[sku]['free_minutes'] += quantity

            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row {row_num} due to error: {e}", file=sys.stderr)
                continue

    return org_sku_data if by_organization else sku_data


def sanitize_filename(name, max_length=100):
    """
    Sanitize a string to be safe for use in a filename.

    Args:
        name: The string to sanitize
        max_length: Maximum length for the sanitized name

    Returns:
        A safe filename string
    """
    if not name:
        return 'unknown'

    # Remove or replace problematic characters
    # Keep only alphanumeric, dash, underscore, and dot
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', name)

    # Remove leading dots or dashes (hidden files, command flags)
    safe = safe.lstrip('.-')

    # Collapse multiple underscores
    safe = re.sub(r'_+', '_', safe)

    # Truncate to max length
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip('_')

    # Ensure it's not empty after sanitization
    return safe if safe else 'unknown'


def sanitize_csv_field(value):
    """
    Sanitize a field to prevent CSV injection attacks.

    Args:
        value: The value to sanitize

    Returns:
        Sanitized string safe for CSV output
    """
    value_str = str(value)
    # If the value starts with a character that could be interpreted as a formula
    if value_str and value_str[0] in ('=', '+', '-', '@', '\t', '\r'):
        # Prefix with a single quote to force text interpretation
        return "'" + value_str
    return value_str


def write_summary_csv(output_file, sku_data):
    """
    Write the summary data to a CSV file.

    Args:
        output_file: Path to the output CSV file
        sku_data: Dictionary with SKU summaries
    """
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)

        # Write header
        writer.writerow(['sku', 'total_net_amount', 'free_minutes_quantity'])

        # Write data sorted by SKU
        for sku in sorted(sku_data.keys()):
            writer.writerow([
                sanitize_csv_field(sku),
                f"{sku_data[sku]['total_cost']:.2f}",
                f"{sku_data[sku]['free_minutes']:.2f}"
            ])


def main():
    parser = argparse.ArgumentParser(
        description='Summarize Actions billing costs by SKU'
    )
    parser.add_argument(
        'input_file',
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--by-org',
        action='store_true',
        help='Also generate separate summary files for each organization'
    )

    args = parser.parse_args()
    input_file = Path(args.input_file)

    if not input_file.exists():
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)

    if not input_file.is_file():
        print(f"Error: '{input_file}' is not a file")
        sys.exit(1)

    try:
        print(f"Processing {input_file}...")

        # Always generate single SKU summary file
        sku_data = process_billing_data(input_file, by_organization=False)
        sku_output_file = input_file.parent / f"{input_file.stem}_sku{input_file.suffix}"
        write_summary_csv(sku_output_file, sku_data)

        print(f"Summary written to {sku_output_file}")
        print(f"Found {len(sku_data)} distinct Actions SKUs")

        if args.by_org:
            # Also generate per-organization files
            org_sku_data = process_billing_data(input_file, by_organization=True)

            # Limit number of organizations to prevent resource exhaustion
            max_orgs = 1000
            if len(org_sku_data) > max_orgs:
                print(f"Warning: Found {len(org_sku_data)} organizations, limiting to {max_orgs}")
                org_sku_data = dict(list(org_sku_data.items())[:max_orgs])

            for organization, org_data in org_sku_data.items():
                # Sanitize organization name for filename
                safe_org_name = sanitize_filename(organization)

                output_file = input_file.parent / f"{input_file.stem}.{safe_org_name}{input_file.suffix}"
                write_summary_csv(output_file, org_data)

            print(f"Generated {len(org_sku_data)} organization-specific files")
            print(f"Found {len(org_sku_data)} distinct organizations")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()