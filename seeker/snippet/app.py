#date: 2021-12-16T17:16:24Z
#url: https://api.github.com/gists/a2e8e81697d9cf2d3bb95b132bbcfd59
#owner: https://api.github.com/users/benyamynbrkyc

import re
import csv
# import os


def sanitize_file(file_contents):
    clean_text = ''

    for idx, line in enumerate(file_contents, start=1):
        identifier = line.strip().split(' ')[:2]
        print(identifier)
        try:
            if isinstance(int(identifier[0]), int) and identifier[1] and identifier[1] == '-':
                clean_text += line + '\n'
        except ValueError:
            pass

        # if idx == 200:
        #     break

    return clean_text


def read_csv(csv_file_path):
    clean = ''
    with open(csv_file_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for r in spamreader:
            row = r[0].strip()
            if not row.startswith(';;'):
                clean += row + '\n'

    return clean


def clean_text(text_path):
    text = []
    clean = ''
    count = 0
    f = open(text_path, 'r')
    for idx, line in enumerate(f, start=0):
        text.append(line)
        found = re.findall(r'([1-9]) -+ +', line)
        # print('found = ', found, ' on line', idx)
        if found:
            _line = ''
            if (' - ' in line):
                _line = line.replace(' - ', '; ')

            if (' -- ' in line):
                _line = line.replace(' -- ', '; ')

            text[idx] = _line

    f.close()

    return text


def save_to_txt(text_arr):
    f = open('out/final.txt', 'w')
    for line in text_arr:
        f.write(line)
    f.close()


def fix():
    out = open('out/*** OVAJ.txt', 'w')
    f = open('out/text-no-empty.txt', 'r')
    outfile = open('csv/out.csv', 'w')

    writer = csv.writer(outfile, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for idx, line in enumerate(f, start=0):
        l = None
        line = line.replace('\n', '')
        _line = line.split(',')
        if len(_line) > 2:
            __line = [_line[0], ''.join(_line[1:])]
            # print('ne valja', idx, _line)
            # print('fixed', idx, __line)
            # print(__line)
            l = __line
        else:
            # print(_line)
            l = _line

        writer.writerow(l)

    f.close()
    out.close()

# def create_csv(txt):
#     for idx, line in enumerate(out, start=0):
#         print(line)


if __name__ == '__main__':
    fix()
