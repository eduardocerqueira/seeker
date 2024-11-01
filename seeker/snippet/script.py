#date: 2024-11-01T17:07:28Z
#url: https://api.github.com/gists/c95186e6cda58e06dcfb86062d58d53f
#owner: https://api.github.com/users/JRoshthen1

import re


strings = set()

with open('texts.txt', mode='r', encoding='utf-8') as file_input:
    content = file_input.read()
    regExStr = r'^\s*\d string title = "(?:tooltip\d+|Name|Dialogue Text)"\n\s*\d string value = "((?!(START|input|\w+\(\)|\!\(\w+\(\)\))).+)"$'
    compiled = re.compile(regExStr, re.MULTILINE)
    matched = compiled.finditer(content)
    for m in matched:
        strings.add(m.groups()[0])

ignored_regex_strings = [
    r'TASK\.',
    r'^\w+\.\w+$',
    r'^\w+_\w+$',
    r'^[a-z]+$',
]

strings_filtered = strings
for regex_string in ignored_regex_strings:
    regex = re.compile(regex_string)
    # strings_filtered = filter(lambda i: not regex.search(i), strings_filtered)
    strings_filtered = [i for i in strings_filtered if not regex.search(i)]

with open('extracted_texts.txt', mode='w', encoding='utf-8') as file_output:
    for string in sorted(strings_filtered):
        file_output.write(f'{string}\n')
