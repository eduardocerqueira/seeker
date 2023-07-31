#date: 2023-07-31T17:08:16Z
#url: https://api.github.com/gists/f9380b6309579c5af55b2a02d80f9901
#owner: https://api.github.com/users/essingen123

import os

def search_keyword_in_file(file_path, keyword):
    findings = []
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines, start=1):
            if keyword.lower() in line.lower():
                findings.append((idx, line.strip()))
    return findings

def search_keyword_in_directory(directory, keyword):
    findings = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.startswith('.'):
                file_path = os.path.join(root, file_name)
                file_findings = search_keyword_in_file(file_path, keyword)
                if file_findings:
                    findings.append((file_path, file_findings))
    return findings

def search_keyword_recursive(directories, keyword):
    all_findings = []
    for directory in directories:
        directory_findings = search_keyword_in_directory(directory, keyword)
        all_findings.extend(directory_findings)
        for file_path, _ in directory_findings:
            if os.path.isdir(file_path):
                sub_directory_findings = search_keyword_recursive([file_path], keyword)
                all_findings.extend(sub_directory_findings)
    return all_findings

def print_findings(all_findings):
    if not all_findings:
        print(f"'{keyword_to_search}' not found in any shell configuration files.")
        return

    for file_path, findings in all_findings:
        print(f"Found '{keyword_to_search}' in {file_path}:")
        for idx, line in findings:
            print(f"  Line {idx}: {line}")

if __name__ == "__main__":
    keyword_to_search = input("Enter the keyword you want to search for: ")
    directories_to_search = ["/etc", os.path.expanduser("~")]

    all_findings = search_keyword_recursive(directories_to_search, keyword_to_search)
    print_findings(all_findings)
