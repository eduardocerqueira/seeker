#date: 2023-03-09T17:02:34Z
#url: https://api.github.com/gists/c8c9fec1b6d2b9eee3eb077aab272a41
#owner: https://api.github.com/users/markshust

import os
import textwrap

input_dir = 'input1/'
output_dir = 'output1/'

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('.srt'):
        with open(input_dir + filename, 'r') as input_file:
            output_filename = output_dir + filename
            with open(output_filename, 'w') as output_file:
                prev_line = ''
                for line in input_file:
                    if line[0].isalpha():
                        try:
                            next_line = next(input_file)
                        except StopIteration:
                             next_line = ''
                        if next_line != '' and next_line[0].isalpha():
                            try:
                                next_next_line = next(input_file)
                            except StopIteration:
                                next_line = ''
                            if next_next_line != '' and next_next_line[0].isalpha():
                                output_file.write(line.strip().replace('\n', '').replace('\r', '') + ' ' + next_line.strip().replace('\n', '').replace('\r', '') + ' ' + next_next_line.strip().replace('\n', '').replace('\r', '')+ '\n')
                            else:
                                output_file.write(line.strip().replace('\n', '').replace('\r', '') + ' ' + next_line.strip().replace('\n', '').replace('\r', '') + '\n\n')
                        else:
                            output_file.write(line + '\n')
                    else:
                        output_file.write(line)

        print(f"Done processing {filename}!")
