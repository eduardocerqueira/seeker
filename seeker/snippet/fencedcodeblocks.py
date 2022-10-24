#date: 2022-10-24T17:14:11Z
#url: https://api.github.com/gists/ef0964e135599b937dfde1de9c868dd2
#owner: https://api.github.com/users/sdarwin

#!/usr/bin/python3

# Replaces indented code with fenced code blocks

import re
import glob

def replfunction1(matchobj):
    matchedstring=matchobj.group(0)
    matchedstring=matchedstring.strip()
    matchedstring=re.sub('^    ', '', matchedstring, flags=re.MULTILINE)
    matchedstring="\n```cpp" + "\n" + matchedstring + "\n" + "```\n\n"
    return(matchedstring)

def replfunction2(matchobj):
    matchedstring=matchobj.group(0)
    print("matched string is \n")
    print(matchedstring)
    matchedstring=matchedstring.strip()
    matchedstring=matchedstring + " "
    matchedstring = re.sub('^/// ', '', matchedstring, flags=re.MULTILINE)
    matchedstring = re.sub('^\s*\n(\s{4}.*$)+\n*', replfunction1, matchedstring, flags=re.MULTILINE)
    matchedstring=matchedstring.strip()
    matchedstring="\n" + matchedstring + "\n"
    matchedstring=re.sub('^', '/// ', matchedstring, flags=re.MULTILINE)
    matchedstring="\n" + matchedstring + "\n\n"
    return(matchedstring)

for filename in glob.glob('**/*.md',recursive=True):
    print("Name is " + filename)
    file = open(filename, 'r+')
    Lines = file.read()
    result = re.sub('^\s*\n(\s{4}.*$)+\n*',replfunction1,Lines,flags=re.MULTILINE)
    file.seek(0)
    file.truncate()
    file.write(result)
    file.close()

for filename in glob.glob('**/*.hpp',recursive=True):
    print("Name is " + filename)
    file = open(filename, 'r+')
    Lines = file.read()
    result = re.sub('^(\s*///.*$)+\n*',replfunction2,Lines,flags=re.MULTILINE)
    file.seek(0)
    file.truncate()
    file.write(result)
    file.close()