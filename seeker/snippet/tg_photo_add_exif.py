#date: 2023-01-23T17:05:11Z
#url: https://api.github.com/gists/8a8ae64e3cb9c3523aa3ac5ca29d3f7b
#owner: https://api.github.com/users/petertrr

#!/bin/env python

import os
import re

name_format = r"photo_(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)_(?P<hour>\d+)-(?P<minute>\d+)-(?P<second>\d+).*\.jpg"
for name in os.listdir("."):
    match = re.match(name_format, name)
    if match is None:
        continue
    date = f"{match.group('year')}:{match.group('month')}:{match.group('day')} " + \
        f"{match.group('hour')}:{match.group('minute')}:{match.group('second')}"
    print(f"Converting file {name}")
    formatted_name = name.replace(' ','\\ ').replace('(', '\\(').replace(')', '\\)')
    os.system(f"exiftool -DateTimeOriginal=\"{date}\" -overwrite_original {formatted_name}")
