#date: 2024-06-06T17:11:40Z
#url: https://api.github.com/gists/2391c1bc7d532725909a46b94e569d60
#owner: https://api.github.com/users/sandro010

import sys

def replace_newline(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
        
    content = content.replace('\\n', '\n')

    with open(output_file, 'w') as file:
        file.write(content)

replace_newline(sys.argv[1], sys.argv[2])