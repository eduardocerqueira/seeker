#date: 2024-08-27T16:54:55Z
#url: https://api.github.com/gists/bae3ba42d696e3d3d597237a680f4dff
#owner: https://api.github.com/users/IMpcuong

#!/usr/bin/env python3
import sys
import json

def parse_value(value):
    try:
        # Try converting to integer
        return int(value)
    except ValueError:
        try:
            # Try converting to float
            return float(value.strip(' C'))  # Also strip ' C' in case of temperatures
        except ValueError:
            # Return original string if no conversion is possible
            return value

def parse_line(line):
    if ':' in line:
        key, value = line.split(':', 1)
        return key.strip(), parse_value(value.strip())
    return None, None

def parse_ssacli_output(ssacli_output):
    data = {'Controller': {}, 'Ports': [], 'Drives': []}
    current_entity = data['Controller']

    for line in ssacli_output:
        line = line.strip()
        if "Port Name" in line:
            current_entity = {}
            data['Ports'].append(current_entity)
        elif "physicaldrive" in line:
            current_entity = {'ID': line.split()[-1]}
            data['Drives'].append(current_entity)
        else:
            key, value = parse_line(line)
            if key:
                current_entity[key] = value

    return data

def main():
    ssacli_output = sys.stdin.readlines()
    parsed_data = parse_ssacli_output(ssacli_output)
    print(json.dumps(parsed_data, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()