#date: 2025-12-19T17:13:52Z
#url: https://api.github.com/gists/431931e9be7ffaad4ff850482ebcdb65
#owner: https://api.github.com/users/sgouda0412

import os
import sys

version = f"{sys.version_info.major}.{sys.version_info.minor}"
print(f"Hello World, from Python {version} in an external script!")
print(f"Did you know that {os.getenv('SPAM_STRING', 'there is a SPAM_STRING')}?")
