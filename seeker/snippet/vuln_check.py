#date: 2025-09-09T17:10:39Z
#url: https://api.github.com/gists/aecacd9af3ca6fc95d85514567f7ee66
#owner: https://api.github.com/users/foad

import json
from pathlib import Path

CODE_DIR = "~/code"

COMPROMISED_PACKAGES = {
    "backslash": "0.2.1",
    "chalk-template": "1.1.1",
    "supports-hyperlinks": "4.1.1",
    "has-ansi": "6.0.1",
    "simple-swizzle": "0.2.3",
    "color-string": "2.1.1",
    "error-ex": "1.3.3",
    "color-name": "2.0.1",
    "is-arrayish": "0.3.3",
    "slice-ansi": "7.1.1",
    "color-convert": "3.1.1",
    "wrap-ansi": "9.0.1",
    "ansi-regex": "6.2.1",
    "supports-color": "10.2.1",
    "strip-ansi": "7.1.1",
    "chalk": "5.6.1",
    "debug": "4.4.2",
    "ansi-styles": "6.2.2",
}

def find_malicious_packages(search_dir):
    search_path = Path(search_dir).expanduser()
    found_count = 0

    print(f"[*] Starting scan in: {search_path}")
    print(f"[*] Searching for {len(COMPROMISED_PACKAGES)} known compromised packages...\n")

    for package_name, malicious_version in COMPROMISED_PACKAGES.items():
        pattern = f"**/node_modules/{package_name}"

        for package_path in search_path.rglob(pattern):
            if not package_path.is_dir():
                continue

            pkg_json_path = package_path / "package.json"
            if not pkg_json_path.is_file():
                continue

            try:
                with open(pkg_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    installed_version = data.get("version")
                    if installed_version == malicious_version:
                        print(f"[!!!] ALERT: Found compromised package '{package_name}' version '{installed_version}' at:")
                        print(f"      -> {package_path}")
                        found_count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"[!] WARNING: Could not read or parse {pkg_json_path}. Error: {e}")


    print("\n[*] Scan complete.")
    if found_count > 0:
        print(f"[*] RESULT: Found {found_count} instance(s) of compromised packages.")
    else:
        print("[*] RESULT: No known compromised packages were found in the specified directory.")

if __name__ == "__main__":
    scan_directory = CODE_DIR
    find_malicious_packages(scan_directory)