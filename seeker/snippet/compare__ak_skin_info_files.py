#date: 2023-10-30T16:47:39Z
#url: https://api.github.com/gists/99abfcb9292b18dbf2fd5249730c8760
#owner: https://api.github.com/users/LCAR979

import pathlib

work_dir = pathlib.Path(__file__).parent.resolve()
f_year = work_dir / 'clothing-by-year.txt'
f_author = work_dir / 'clothing-by-author-y.txt'

with open(f_year,'r', encoding='utf8') as f:
    content = f.readlines()
    lines_year = [l[5:-1] for l in content if l.startswith('|时装名')]
    print(f"year file count: {len(lines_year)}")
    set_year = set(lines_year)

with open(f_author,'r', encoding='utf8') as f:
    content = f.readlines()
    lines_author = [l[5:-1] for l in content if l.startswith('|时装名')]
    print(f"author file count: {len(lines_author)}")
    set_author = set(lines_author)

print("set difference: ")
print(set_year - set_author)
print(set_author - set_year)