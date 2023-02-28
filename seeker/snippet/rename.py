#date: 2023-02-28T16:50:13Z
#url: https://api.github.com/gists/c823595581f97528cf295598194f8cf1
#owner: https://api.github.com/users/brian-bacch

"""Programmatically rename files based on pattern matching."""

from glob import iglob
from os import rename
from os.path import join, split

def rename_stuff(indir, replaced, replace_with):
    count = 0
    for fname in iglob(join(indir, f'*{replaced}*')):
        new_fname = join(indir, split(fname)[1].replace(replaced, replace_with))
        rename(fname, new_fname)
        count += 1
    print(f'Renamed {count} items')


if __name__ == "__main__":
    from sys import argv

    if '-h' in argv or '--help' in argv:
        print('python rename.py <indir> <replaced> <replacewith>')
        quit()

    indir = argv[1]
    replaced = argv[2]
    replace_with = argv[3]
    
    rename_stuff(indir, replaced, replace_with)
