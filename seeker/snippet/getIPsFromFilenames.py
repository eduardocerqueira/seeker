#date: 2025-03-11T16:55:17Z
#url: https://api.github.com/gists/937a13ccad0a618f8253f2c6dce71f78
#owner: https://api.github.com/users/and-kal

import os
import re

def extract_ip_from_filename(filename):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    match = re.search(ip_pattern, filename)
    return match.group(0) if match else None

def scan_folder_for_ips(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return
    
    ip_addresses = set()
    
    for filename in os.listdir(folder_path):
        ip = extract_ip_from_filename(filename)
        if ip:
            ip_addresses.add(ip)
    
    return ip_addresses

def main():
    folder_path = input("Enter the folder path: ")
    extracted_ips = scan_folder_for_ips(folder_path)
    
    if extracted_ips:
        for ip in extracted_ips:
            print(ip)

if __name__ == "__main__":
    main()
