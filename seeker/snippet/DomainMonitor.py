#date: 2025-02-21T16:50:32Z
#url: https://api.github.com/gists/c87d7fc46041945bd0717714261752f8
#owner: https://api.github.com/users/TechByTom

import sys
import os
import json
import subprocess
import argparse
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import pickle

# Configuration
INPUT_FILE = 'ListOfCompanyOwnedDomains.txt'
OUTPUT_DIR = 'rawOutput'
DNSTWIST_PATH = 'dnstwist'
ALERT_EMAILS = 'youremailhere@email.domain'
SMTP_SERVER = 'smtp.email.domain'
SMTP_PORT = 25
SENDER_EMAIL = 'youremailhere@email.domain'

# Colors
RED = '\033[91m'
RESET = '\033[0m'

# Ensure OUTPUT_DIR exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

def print_status(message, color=RESET):
    print(f"{color}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}{RESET}")

def run_dnstwist(domain):
    print_status(f"Running dnstwist for domain: {domain}")
    try:
        output = subprocess.check_output([DNSTWIST_PATH, domain, '-r', '-m', '-w', '--format', 'json'], stderr=subprocess.STDOUT)
        results = json.loads(output)

        # Print registered domains immediately
        registered_domains = [r for r in results if isinstance(r, dict) and 'domain' in r]
        if registered_domains:
            print_status("Registered lookalike domains found:")
            for r in registered_domains:
                print_status(f"  → {r['domain']}", color=RED)
                if 'dns_a' in r and r['dns_a']:
                    print_status(f"    IP: {', '.join(r['dns_a'])}")
                if 'whois_created' in r:
                    print_status(f"    Registered: {r['whois_created']}")
                if 'dns_mx' in r and r['dns_mx']:
                    print_status(f"    MX: {', '.join(r['dns_mx'])}")

        print_status(f"Found {len(results)} registered lookalike domains for {domain}")
        return results
    except subprocess.CalledProcessError as e:
        print_status(f"{RED}Error running dnstwist: {e.output.decode()}{RESET}")
        return []
    except FileNotFoundError:
        print_status(f"{RED}Error: dnstwist not found. Make sure it's installed and in your PATH.{RESET}")
        print("You can install dnstwist using: pip install dnstwist")
        sys.exit(1)

def save_results(domain, results):
    """Save current scan results and return the file path."""
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{domain}_results_{date_str}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print_status(f"Current scan results saved to {os.path.basename(output_file)}")
    return output_file

def load_previous_results(domain, current_file=None):
    """Load the most recent previous results file for comparison, excluding the current scan."""
    domain_files = []

    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.startswith(f'{domain}_results_') and file.endswith('.json'):
                file_path = os.path.join(root, file)
                if current_file and os.path.abspath(file_path) == os.path.abspath(current_file):
                    continue
                domain_files.append(file_path)

    if not domain_files:
        print_status(f"No previous scan results found for domain: {domain}")
        return None

    domain_files.sort(key=os.path.getmtime, reverse=True)
    most_recent_file = domain_files[0]

    print_status(f"Using most recent scan from {os.path.basename(most_recent_file)} for comparison")

    try:
        with open(most_recent_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print_status(f"{RED}Error loading previous results: {str(e)}{RESET}")
        return None

def load_reported_domains():
    try:
        with open('reported_domains.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return set()

def save_reported_domains(domains):
    with open('reported_domains.pkl', 'wb') as f:
        pickle.dump(domains, f)

def compare_results(domain, current_results, previous_results, reported_domains, input_domains):
    print_status(f"Comparing results for domain: {domain}")
    changes = []

    current_results = [r for r in current_results if isinstance(r, dict) and 'domain' in r]

    if previous_results is None:
        print_status(f"{RED}No previous scan results found - this appears to be the first run{RESET}")
        changes = [
            {'domain': d['domain'], 'reason': 'initial detection', 'details': d}
            for d in current_results
            if d['domain'] not in reported_domains
            and d.get('fuzzer') != '*original'
            and d['domain'] not in input_domains
        ]
        if changes:
            print_status(f"First run - detected {len(changes)} domains to report")
    else:
        previous_domains = {d['domain']: d for d in previous_results if isinstance(d, dict) and 'domain' in d}

        for current in current_results:
            if (current['domain'] in reported_domains
                or current.get('fuzzer') == '*original'
                or current['domain'] in input_domains):
                continue

            previous = previous_domains.get(current['domain'])

            if not previous:
                changes.append({'domain': current['domain'], 'reason': 'new domain', 'details': current})
                print_status(f"New domain detected: {current['domain']}")
            else:
                if (current.get('whois_created') != previous.get('whois_created') or
                    current.get('whois_registrar') != previous.get('whois_registrar')):
                    changes.append({'domain': current['domain'], 'reason': 'registration change', 'details': current})
                    print_status(f"Registration change detected: {current['domain']}")
                elif bool(current.get('dns_mx')) != bool(previous.get('dns_mx')):
                    changes.append({'domain': current['domain'], 'reason': 'MX record change', 'details': current})
                    print_status(f"MX record change detected: {current['domain']}")

    print_status(f"Found {len(changes)} changes for {domain}")
    return changes

def generate_report_content(domains):
    """Generate the report content as a string."""
    body = "The following lookalike domains have been detected as new or changed:\n\n"
    
    # Group domains by their original monitored domain
    grouped_domains = {}
    for domain in domains:
        original_domain = domain['details'].get('original', 'unknown')
        if original_domain not in grouped_domains:
            grouped_domains[original_domain] = []
        grouped_domains[original_domain].append(domain)
    
    # Generate report for each group
    for original_domain, domain_group in grouped_domains.items():
        body += f"\nResults for monitored domain: {original_domain}\n"
        body += "-" * 50 + "\n"
        for domain in domain_group:
            details = domain['details']
            body += f"Domain: {details['domain']}\n"
            body += f"Reason for Alert: {domain['reason']}\n"
            body += f"A Records: {', '.join(details.get('dns_a', []))}\n"
            body += f"NS Records: {', '.join(details.get('dns_ns', []))}\n"
            body += f"MX Records: {', '.join(details.get('dns_mx', []))}\n"
            if 'whois_created' in details:
                body += f"Registration Date: {details['whois_created']}\n"
            if 'whois_registrar' in details:
                body += f"Registrar: {details['whois_registrar']}\n"
            body += "\n"
    return body

def send_alert(domains, report_file=None):
    """Send alert via email or write to file based on configuration."""
    print_status("Preparing alert")
    report_content = generate_report_content(domains)

    if report_file:
        # Write to file
        try:
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report_content)
            print_status(f"Alert report written to: {report_file}")
        except Exception as e:
            print_status(f"{RED}Error writing report to file: {str(e)}{RESET}")
            print_status(f"{RED}Report content:{RESET}\n\n{report_content}")
    else:
        # Send email
        msg = MIMEMultipart()
        msg['From'] = formataddr(("DNS Twist Alert", SENDER_EMAIL))
        msg['To'] = ALERT_EMAILS
        msg['Subject'] = 'URGENT: New or Changed Lookalike Domains Detected'
        msg['Importance'] = 'high'
        msg['X-Priority'] = '1'

        msg.attach(MIMEText(report_content, 'plain'))

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.send_message(msg)
            server.quit()
            print_status("Alert email sent successfully")
        except Exception as e:
            print_status(f"{RED}Warning: SMTP server is unavailable. Alert email could not be sent.{RESET}")
            print_status(f"{RED}Email content:{RESET}\n\n{report_content}")

def main():
    parser = argparse.ArgumentParser(description='DNS lookalike domain monitoring tool')
    parser.add_argument('--report-file', type=str, help='Write report to file instead of sending email')
    args = parser.parse_args()

    print_status("Starting domain lookalike check")
    try:
        with open(INPUT_FILE, 'r') as f:
            input_domains = [d.strip().lower() for d in f.read().splitlines() if d.strip()]
    except FileNotFoundError:
        print_status(f"{RED}Error: Input file {INPUT_FILE} not found{RESET}")
        sys.exit(1)

    print_status(f"Loaded {len(input_domains)} domains from {INPUT_FILE}")

    reported_domains = load_reported_domains()
    all_changed_domains = []  # New list to collect all changes

    for domain in input_domains:
        print_status(f"Processing domain: {domain}")
        current_results = run_dnstwist(domain)

        if not current_results:
            print_status(f"{RED}Warning: No results returned for {domain}{RESET}")
            continue

        current_file = save_results(domain, current_results)
        previous_results = load_previous_results(domain, current_file)
        changed_domains = compare_results(domain, current_results, previous_results, reported_domains, input_domains)

        if changed_domains:
            print_status(f"Changes detected for domain: {domain}")
            # Add the original domain to each result for grouping in the report
            for change in changed_domains:
                change['details']['original'] = domain
            all_changed_domains.extend(changed_domains)
            reported_domains.update(d['details']['domain'] for d in changed_domains)
        else:
            print_status(f"No changes detected for domain: {domain}")

    # Send alert only once with all changes
    if all_changed_domains:
        print_status(f"Generating alert for {len(all_changed_domains)} total changes")
        send_alert(all_changed_domains, args.report_file)

    save_reported_domains(reported_domains)
    print_status("Domain lookalike check completed")

if __name__ == '__main__':
    main()