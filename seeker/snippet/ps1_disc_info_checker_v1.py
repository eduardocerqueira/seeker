#date: 2025-04-25T16:39:57Z
#url: https://api.github.com/gists/b6f3d8daf8a41b5ceec9258de1c336a5
#owner: https://api.github.com/users/EncodeTheCode

import os
import re
import time
import platform

def check_cd_drive_ready(cd_path):
    """Check if the CD drive is accessible and ready."""
    if not os.path.exists(cd_path):
        print(f"❌ Drive {cd_path} not found.")
        return False
    if not os.path.isdir(cd_path):
        print(f"❌ {cd_path} is not a valid directory.")
        return False
    return True

def read_system_cnf(cd_path):
    """Read SYSTEM.CNF from the PS1 game disc to extract serial number."""
    system_cnf_path = os.path.join(cd_path, 'SYSTEM.CNF')
    if not os.path.exists(system_cnf_path):
        return None, "SYSTEM.CNF not found on the disc."

    try:
        with open(system_cnf_path, 'r', errors='ignore') as file:
            content = file.read()
            match = re.search(r'BOOT\s*=\s*cdrom:\\\\?([^;]+);', content, re.IGNORECASE)
            if match:
                serial = match.group(1).replace('.', '').strip()
                return serial, None
            else:
                return None, "Serial number not found in SYSTEM.CNF."
    except Exception as e:
        return None, f"Error reading SYSTEM.CNF: {e}"

def extract_strings_from_files(cd_path, max_files=3):
    """Extract readable strings from the first few files on the disc."""
    strings_found = []
    try:
        entries = os.listdir(cd_path)
        entries = [f for f in entries if os.path.isfile(os.path.join(cd_path, f))]
        for file in entries[:max_files]:
            path = os.path.join(cd_path, file)
            try:
                with open(path, 'rb') as f:
                    data = f.read(1024 * 1024)  # Read first 1MB
                    strings = re.findall(rb'[\x20-\x7E]{4,}', data)
                    for s in strings:
                        decoded = s.decode('ascii', errors='ignore')
                        strings_found.append(decoded)
            except Exception as e:
                strings_found.append(f"[Error reading {file}: {e}]")
    except Exception as e:
        strings_found.append(f"[Error listing files: {e}]")
    return strings_found

def scan_ps1_disc(cd_path="F:/"):
    """Main function to scan the PS1 disc for game information."""
    print(f"🔍 Using fixed CD-ROM drive: {cd_path}")
    
    # Ensure the CD drive is accessible and ready
    if not check_cd_drive_ready(cd_path):
        return {"errors": [f"Drive {cd_path} is not ready or accessible."]}

    print("⏳ Grace period... Spinning up disc.")
    time.sleep(5)

    # Result dictionary to store all relevant data
    result = {
        "os_platform": platform.system(),
        "disc_device": cd_path,
        "serial_number": "Unknown",
        "possible_game_title": "Unknown",
        "errors": [],
        "extracted_strings_sample": []
    }

    # Try to extract the serial number from SYSTEM.CNF
    serial, error = read_system_cnf(cd_path)
    if serial:
        result["serial_number"] = serial
    else:
        result["errors"].append(error)

    # Extract readable strings from the first few files
    strings = extract_strings_from_files(cd_path)
    result["extracted_strings_sample"] = strings
    title_guess = next((s for s in strings if 'title' in s.lower()), None)
    if title_guess:
        result["possible_game_title"] = title_guess

    return result

# Run script
if __name__ == "__main__":
    data = scan_ps1_disc()

    # Print the results of the scan
    print("\n📀 PS1 Disc Scan Result:")
    for key, value in data.items():
        if key == "extracted_strings_sample":
            print(f"{key} (showing {min(len(value), 10)} strings):")
            for line in value[:10]:
                print("  ", line)
        elif key == "errors" and value:
            print("⚠️ Errors:")
            for err in value:
                print("  -", err)
        else:
            print(f"{key}: {value}")
