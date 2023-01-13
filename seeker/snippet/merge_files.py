#date: 2023-01-13T16:42:57Z
#url: https://api.github.com/gists/f0ea0686478eddd2eec0d1193d277785
#owner: https://api.github.com/users/AdrianAcala

import sys

def merge_files(file_list):
    # open all files
    files = [open(f, 'r') for f in file_list]

    # create a list of lines from all files
    lines = [line for file in files for line in file]
    # close all files
    [file.close() for file in files]

    # concatenate lines starting with whitespace to the preceding line
    i = 0
    while i < len(lines) - 1:
        if lines[i + 1][0].isspace():
            lines[i] = lines[i] + lines[i + 1]
            del lines[i + 1]
        else:
            i += 1

    # sort lines alphabetically
    lines.sort()

    # write the merged and sorted lines to a new file
    with open('merged_alphabetical.txt', 'w') as f:
        for line in lines:
            f.write(line)

if __name__ == '__main__':
    merge_files(sys.argv[1:])
