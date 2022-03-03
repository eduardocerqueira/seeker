#date: 2022-03-03T17:04:56Z
#url: https://api.github.com/gists/d15efd17d7311ff81eae2b50d8f72f48
#owner: https://api.github.com/users/guparan

import os
import sys

version = f"{sys.version_info.major}.{sys.version_info.minor}"
print(f"Hello World, from Python {version} in an external script!")
print(f"Did you know that {os.getenv('SPAM_STRING', 'there is a SPAM_STRING')}?")
