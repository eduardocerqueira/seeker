#date: 2023-03-29T17:26:23Z
#url: https://api.github.com/gists/ff6821a017fc52a6a9f950b9e6a375e2
#owner: https://api.github.com/users/erfanoabdi

import xml.etree.ElementTree as ET

# Parse the compatibility matrix XML
compatibility_matrix = ET.parse('<path to compatibility_matrix xml>')

# Extract the kernel version and required configs from the XML
kernel_version = '4.19.191'
required_configs = []
for kernel in compatibility_matrix.findall('.//kernel'):
    if kernel.attrib['version'] == kernel_version:
        for config in kernel.findall('config'):
            value = config.find('value').text
            if value == 'y':
                required_configs.append(config.find('key').text)

# Load the defconfig file
defconfig_file = '<path to defconfig>'
with open(defconfig_file, 'r') as f:
    defconfig = f.read()

# Check for configs in conditions
for kernel in compatibility_matrix.findall('.//kernel'):
    if kernel.attrib['version'] == kernel_version:
        has_condition = False
        for condition in kernel.findall('conditions/config'):
            key = condition.find('key').text
            value = condition.find('value').text
            if f'{key}={value}' in defconfig:
                has_condition = True
        if has_condition:
            for config in kernel.findall('config'):
                value = config.find('value').text
                if value == 'y':
                    required_configs.append(config.find('key').text)

# Check for missing configs
missing_configs = []
for config in required_configs:
    if f'{config}=n' in defconfig or f'# {config} is not set' in defconfig:
        missing_configs.append(config)

# Print the results
if missing_configs:
    print('Missing configs:')
    for config in missing_configs:
        print(config)
else:
    print('All required configs are present')

