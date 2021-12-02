#date: 2021-12-02T17:14:18Z
#url: https://api.github.com/gists/763a4aeadcaab637ce8360b8df046f7c
#owner: https://api.github.com/users/Avielyo10

import re
import inflection
import in_place
import glob

prog = re.compile('(.*)(json:")([^,"]*)(.*)')

for goFile in glob.glob("*/**/*.go"):
    print(goFile)
    with in_place.InPlace(goFile) as inFile:
        for line in inFile:
            if prog.match(line):
                to_snack_case = prog.sub(r'\3', line)
                to_snack_case = inflection.underscore(to_snack_case).strip()
                new_line = prog.sub(r'\1\2'+to_snack_case+r'\4', line)
                inFile.write(new_line)
            else:
                inFile.write(line)
