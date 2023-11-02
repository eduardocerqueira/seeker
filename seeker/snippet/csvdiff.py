#date: 2023-11-02T16:42:31Z
#url: https://api.github.com/gists/013972939f60a85a2ffea044dfa75139
#owner: https://api.github.com/users/vadimkantorov

# python csvdiff.py mycsv1.csv - mycsv2.csv --col1 "Email" --col2 "Email acheteur" > setdiff.csv

import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file1')
parser.add_argument('op', choices = ['-'])
parser.add_argument('file2')
parser.add_argument('--col1', default = 'email')
parser.add_argument('--col2', default = 'email')
args = parser.parse_args()

col = lambda rows, fieldname: ([k for k in rows[0] if k.lower() == fieldname.lower()] + [None])[0] if rows else None

print(args, file = sys.stderr)
reader1 = csv.DictReader(open(args.file1))
reader2 = csv.DictReader(open(args.file2))
rows1 = list(reader1)
rows2 = list(reader2)
col1 = col(rows1, args.col1)
col2 = col(rows2, args.col2)
print('col1:', col1, file = sys.stderr)
print('col2:', col2, file = sys.stderr)
print('sorted(reader1.fieldnames):', sorted(reader1.fieldnames), file = sys.stderr)
print('sorted(reader2.fieldnames):', sorted(reader2.fieldnames), file = sys.stderr)

if col1 and col2:
    vals = set(d[col2].lower() for d in rows2)
    rows = [d for d in rows1 if d[col1].lower() not in vals]

    writer = csv.DictWriter(sys.stdout, fieldnames = reader1.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
~