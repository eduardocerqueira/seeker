#date: 2023-10-04T17:03:48Z
#url: https://api.github.com/gists/7bc194a8fb2664c25ddfe89df5af156b
#owner: https://api.github.com/users/gbts

#!/usr/bin/env python3
import fileinput
import glob
import os
import re
import sys

pattern = re.compile(r'from [\'\"](\.(\.)?(\/)?.*)[\'\"];')

def regex_match_rewrite(match, file_path, debug=False):
  dirname = os.path.dirname(file_path)
  if match.group(1).endswith('.js'):
    return match.group(0)

  new_import = "from '{0}.js';".format(match.group(1))
  if os.path.isdir(os.path.join(dirname, match.group(1))):
    new_import = "from '{0}/index.js';".format(match.group(1))

  if debug:
    print(match.group(0))
    print(new_import)
  return new_import

def replace_all_ts(commit=False):
  for filename in glob.iglob('**/*.ts', recursive=True):
    dirname = os.path.dirname(filename)
    with fileinput.FileInput(filename, inplace=commit) as file:
      for line in file:
        if not commit:
          pattern.sub(
            lambda match: regex_match_rewrite(match, filename, debug=True),
            line
          )
        else:
          sys.stdout.write(
            pattern.sub(
              lambda match: regex_match_rewrite(match, filename),
              line
            )
          )

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == '--apply':
    replace_all_ts(True)
  else:
    replace_all_ts()