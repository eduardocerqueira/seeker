#date: 2023-03-02T16:48:37Z
#url: https://api.github.com/gists/52cdf9d2cdec6a534318c689346ff478
#owner: https://api.github.com/users/camarokris

import csv
import socket
import urllib.parse
from ipaddress import ip_address
from datetime import datetime
import sys
import threading
from tqdm import tqdm

def is_valid_ipv4_address(ip_address_str):
    try:
        return ip_address(ip_address_str).version == 4
    except ValueError:
        return False

def get_ip_addresses(hostname):
    try:
        return list(
            set([str(ip[4][0]) for ip in socket.getaddrinfo(hostname, None, socket.AF_INET)] + [hostname.lower()]))
    except socket.gaierror:
        return []

def lookup_dns_names(ip_address):
    try:
        return socket.gethostbyaddr(ip_address)[0]
    except socket.herror:
        return None

def process_row(row, writer, progress_bar=None):
    ip_dns = row['IP/DNS']
    port = row['Port']
    if is_valid_ipv4_address(ip_dns):
        row['Hosts'] = str(ip_dns)
        revip = lookup_dns_names(str(ip_dns))
        if revip:
            row['Found FQDN'] = revip
        else:
            return
        if port:
            del row['IP/DNS']
            writer.writerow(row)
        else:
            return
    elif '//' in ip_dns:
        url = urllib.parse.urlparse(ip_dns)
        hostname = url.hostname
        if hostname:
            hosts = get_ip_addresses(hostname)
            hosts.sort(reverse=True)
            if hosts:
                if not port:
                    port = url.port or (80 if url.scheme.lower() == 'http' else 443)
                    row['Port'] = port
                del row['IP/DNS']
                for i in hosts:
                    row['Hosts'] = str(i)
                    writer.writerow(row)
    else:
        hosts = get_ip_addresses(ip_dns)
        hosts.sort(reverse=True)
        if hosts:
            if not port:
                return
            else:
                del row['IP/DNS']
                for i in hosts:
                    row['Hosts'] = str(i)
                    writer.writerow(row)
    if progress_bar:
        progress_bar.update(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py input_file.csv num_threads")
        sys.exit(1)
    input_file_path = sys.argv[1]
    num_threads = int(sys.argv[2])
    output_file_prefix = 'Initiate_Formatted_'
    current_datetime = datetime.now().strftime('%m%d%y.%H%M%S')
    output_file_path = f"{output_file_prefix}{current_datetime}.csv"
    with open(input_file_path, newline='') as input_file, open(output_file_path, 'w', newline='') as output_file:
        reader = csv.DictReader(input_file)
        headers = reader.fieldnames
        nhead = [value for value in headers if value != 'IP/DNS']
        nhead.remove('Port')
        writer = csv.DictWriter(output_file, fieldnames=nhead + ['Hosts', 'Port', 'Found FQDN'])
        writer.writeheader()
        rows = [row for row in reader]
        chunk_size = len(rows) // num_threads
        chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
        threads = []
        with tqdm(total=len(rows)) as pbar:
            for i in range(num_threads):
                thread_rows = chunks[i]
                thread = threading.Thread(target=lambda rows: [process_row(row, writer, progress_bar=pbar) for row in rows], args=(thread_rows,))
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()